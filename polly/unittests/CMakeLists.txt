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

    target_link_libraries(${test_name} gtest_main gtest)
    add_dependencies(PollyUnitTests ${test_name})

    set_property(TARGET ${test_name} PROPERTY FOLDER "Polly")
  endif()
  target_link_libraries(${test_name} Polly LLVMCore LLVMSupport LLVMDemangle LLVMipo)
endfunction()

add_subdirectory(Isl)
add_subdirectory(Flatten)
add_subdirectory(DeLICM)
