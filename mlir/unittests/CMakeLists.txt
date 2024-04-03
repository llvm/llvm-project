set_target_properties(MLIRUnitTests PROPERTIES FOLDER "MLIR Tests")

# To silence warning caused by Wundef.
add_definitions(-DGTEST_NO_LLVM_SUPPORT=0)

function(add_mlir_unittest test_dirname)
  add_unittest(MLIRUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Analysis)
add_subdirectory(Bytecode)
add_subdirectory(Conversion)
add_subdirectory(Debug)
add_subdirectory(Dialect)
add_subdirectory(Interfaces)
add_subdirectory(IR)
add_subdirectory(Parser)
add_subdirectory(Pass)
add_subdirectory(Support)
add_subdirectory(Rewrite)
add_subdirectory(TableGen)
add_subdirectory(Target)
add_subdirectory(Transforms)

if(MLIR_ENABLE_EXECUTION_ENGINE)
  add_subdirectory(ExecutionEngine)
endif()
