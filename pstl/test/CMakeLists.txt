#===-- CMakeLists.txt ----------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===##

# TODO(ldionne): This CMake testing infrastructure should be replaced with a
#                llvm-lit test suite.

add_custom_target(pstl-build-tests
    COMMENT "Build all the pstl tests.")

add_custom_target(check-pstl
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure
    USES_TERMINAL
    DEPENDS pstl-build-tests
    COMMENT "Build and run all the unit tests.")

file(GLOB_RECURSE UNIT_TESTS "test_*.cpp")
foreach(_file IN LISTS UNIT_TESTS)
    file(RELATIVE_PATH _target "${CMAKE_CURRENT_SOURCE_DIR}" "${_file}")
    string(REPLACE ".cpp" "" _target "${_target}")
    set(_target "pstl-${_target}")

    add_executable(${_target} EXCLUDE_FROM_ALL "${_file}")
    target_link_libraries(${_target} PRIVATE pstl::ParallelSTL)
    set_target_properties(${_target} PROPERTIES CXX_EXTENSIONS NO
                                                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    add_test(${_target} "${CMAKE_CURRENT_BINARY_DIR}/${_target}")
    add_dependencies(pstl-build-tests ${_target})
endforeach()
