#===-- CMakeLists.txt ----------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

add_library(test_stdlib INTERFACE)
target_include_directories(test_stdlib INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/support/stdlib")
target_link_libraries(test_stdlib INTERFACE pstl::ParallelSTL)
target_compile_options(test_stdlib INTERFACE -Wno-gnu-include-next)

file(GLOB_RECURSE UNIT_TESTS "*.pass.cpp")
foreach(_file IN LISTS UNIT_TESTS)
    file(RELATIVE_PATH _target "${CMAKE_CURRENT_SOURCE_DIR}" "${_file}")
    string(REPLACE ".cpp" "" _target "${_target}")
    string(REPLACE "/" "-" _target "${_target}")
    set(_target "pstl-${_target}")

    add_executable(${_target} EXCLUDE_FROM_ALL "${_file}")
    target_include_directories(${_target} PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
    target_compile_options(${_target} PRIVATE -Wno-unused-local-typedef -Wno-unused-variable)
    target_link_libraries(${_target} PRIVATE test_stdlib)
    set_target_properties(${_target} PROPERTIES CXX_EXTENSIONS NO
                                                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    add_test(${_target} "${CMAKE_CURRENT_BINARY_DIR}/${_target}")
    add_dependencies(pstl-build-tests ${_target})
endforeach()
