set_target_properties(MLIRUnitTests PROPERTIES FOLDER "MLIR Tests")

function(add_mlir_unittest test_dirname)
  add_unittest(MLIRUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Analysis)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Interfaces)
add_subdirectory(IR)
add_subdirectory(Pass)
add_subdirectory(Support)
add_subdirectory(Rewrite)
add_subdirectory(TableGen)
add_subdirectory(Transforms)

# The native target may not be enabled when cross compiling.
if(TARGET ${LLVM_NATIVE_ARCH})
  add_subdirectory(ExecutionEngine)
endif()
