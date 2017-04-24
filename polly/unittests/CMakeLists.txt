add_custom_target(PollyUnitTests)
set_target_properties(PollyUnitTests PROPERTIES FOLDER "Polly")

# add_polly_unittest(test_dirname file1.cpp file2.cpp)
#
# Will compile the list of files together and link against Polly and its dependences.
function(add_polly_unittest test_name)
  if(COMMAND add_unittest)
    add_unittest(PollyUnitTests ${test_name} ${ARGN})
    target_link_libraries(${test_name} Polly)

    # The Polly target does not depend on its required libraries, except:
    # - BUILD_SHARED_LIBS=ON
    #     in which case calling target_link_libraries again is redundant.
    # - LLVM_LINK_LLVM_DYLIB=ON
    #     in which case it depends on libLLVM.so, so no additional libs needed.
    #     We are also not allowed to link against the plain LLVM* libraries,
    #     which would result in multiple instances of these to be loaded.
    if (NOT LLVM_LINK_LLVM_DYLIB)
      target_link_libraries(${test_name} LLVMCore LLVMSupport LLVMDemangle LLVMipo)
    endif ()
  else()
    add_executable(${test_name} EXCLUDE_FROM_ALL ${ARGN})
    set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    target_link_libraries(${test_name} gtest_main gtest Polly ${LLVM_LIBS})
    add_dependencies(PollyUnitTests ${test_name})

    set_property(TARGET ${test_name} PROPERTY FOLDER "Polly")
  endif()
endfunction()

add_subdirectory(Isl)
add_subdirectory(Flatten)
add_subdirectory(DeLICM)
