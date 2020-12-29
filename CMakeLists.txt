cmake_minimum_required(VERSION 3.17)
project("SLAM_diapackage")

find_package(OpenCV 3.0 REQUIRED)


set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS})
add_subdirectory(${PROJECT_SOURCE_DIR}/test)
