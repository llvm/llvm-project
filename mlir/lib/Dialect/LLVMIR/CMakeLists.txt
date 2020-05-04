add_subdirectory(Transforms)

add_mlir_dialect_library(MLIRLLVMIR
  IR/LLVMDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  DEPENDS
  MLIRLLVMOpsIncGen
  MLIRLLVMConversionsIncGen

  LINK_COMPONENTS
  AsmParser
  BitReader
  BitWriter
  Core
  FrontendOpenMP

  LINK_LIBS PUBLIC
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIROpenMP
  MLIRIR
  MLIRSideEffects
  MLIRSupport
  )

add_mlir_dialect_library(MLIRLLVMAVX512
  IR/LLVMAVX512Dialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  DEPENDS
  MLIRLLVMAVX512IncGen
  MLIRLLVMAVX512ConversionsIncGen

  LINK_COMPONENTS
  AsmParser
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMIR
  MLIRSideEffects
  )

add_mlir_dialect_library(MLIRNVVMIR
  IR/NVVMDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  DEPENDS
  MLIRNVVMOpsIncGen
  MLIRNVVMConversionsIncGen

  LINK_COMPONENTS
  AsmParser
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMIR
  MLIRSideEffects
  )

add_mlir_dialect_library(MLIRROCDLIR
  IR/ROCDLDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  DEPENDS
  MLIRROCDLOpsIncGen
  MLIRROCDLConversionsIncGen

  LINK_COMPONENTS
  AsmParser
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffects
  MLIRVectorToLLVM
  )
