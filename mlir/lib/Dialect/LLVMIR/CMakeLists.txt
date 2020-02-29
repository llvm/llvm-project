add_mlir_dialect_library(MLIRLLVMIR
  IR/LLVMDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  LINK_LIBS
  LLVMAsmParser
  LLVMCore
  LLVMSupport
  MLIRIR
  MLIRSupport
  )
add_dependencies(MLIRLLVMIR
  MLIRLLVMOpsIncGen
  MLIRLLVMConversionsIncGen
  )

add_mlir_dialect_library(MLIRNVVMIR
  IR/NVVMDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  LINK_LIBS
  LLVMAsmParser
  MLIRIR
  MLIRLLVMIR
  LLVMSupport
  LLVMCore
  )
add_dependencies(MLIRNVVMIR
  MLIRNVVMOpsIncGen
  MLIRNVVMConversionsIncGen
  )



add_mlir_dialect_library(MLIRROCDLIR
  IR/ROCDLDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  LINK_LIBS
  LLVMAsmParser
  LLVMCore
  LLVMSupport
  MLIRIR
  )
add_dependencies(MLIRROCDLIR
  MLIRROCDLOpsIncGen
  MLIRROCDLConversionsIncGen
  )
