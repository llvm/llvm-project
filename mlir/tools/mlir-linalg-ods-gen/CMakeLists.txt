set(LLVM_LINK_COMPONENTS
  Core
  Support
  )
add_mlir_tool(mlir-linalg-ods-gen
  mlir-linalg-ods-gen.cpp
)
llvm_update_compile_flags(mlir-linalg-ods-gen)
target_link_libraries(mlir-linalg-ods-gen PRIVATE
  MLIRSupport
  MLIRIR
  )
