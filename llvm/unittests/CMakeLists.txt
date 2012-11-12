add_custom_target(UnitTests)
set_target_properties(UnitTests PROPERTIES FOLDER "Tests")

function(add_llvm_unittest test_dirname)
  add_unittest(UnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(ADT)
add_subdirectory(Analysis)
add_subdirectory(ExecutionEngine)
add_subdirectory(Bitcode)
add_subdirectory(Support)
add_subdirectory(Transforms)
add_subdirectory(VMCore)
