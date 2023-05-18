add_custom_target(PollyUnitTests)
set_target_properties(PollyUnitTests PROPERTIES FOLDER "Polly")

# add_polly_unittest(test_dirname file1.cpp file2.cpp)
#
# Will compile the list of files together and link against Polly and its dependences.
function(add_polly_unittest test_name)
  if(COMMAND add_unittest)
    add_unittest(PollyUnitTests ${test_name} ${ARGN})
  else()
    add_executable(${test_name} EXCLUDE_FROM_ALL ${ARGN})
    set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    target_link_libraries(${test_name} PRIVATE gtest_main gtest)
    add_dependencies(PollyUnitTests ${test_name})

    set_property(TARGET ${test_name} PROPERTY FOLDER "Polly")
  endif()

  if(LLVM_LINK_LLVM_DYLIB AND LLVM_POLLY_LINK_INTO_TOOLS)
    # In this case Polly is already present in libLLVM,
    # no need to link it again.
  else()
    target_link_libraries(${test_name} PRIVATE Polly)
  endif()
endfunction()

add_subdirectory(Isl)
add_subdirectory(Flatten)
add_subdirectory(DeLICM)
add_subdirectory(ScopPassManager)
add_subdirectory(ScheduleOptimizer)
add_subdirectory(Support)
