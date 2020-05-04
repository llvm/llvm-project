add_llvm_tool(mlir-linalg-ods-gen
  mlir-linalg-ods-gen.cpp
)
llvm_update_compile_flags(mlir-linalg-ods-gen)
target_link_libraries(mlir-linalg-ods-gen PRIVATE
  MLIRParser
  MLIRSupport
  LLVMCore
  LLVMSupport
  )
