add_subdirectory(Transforms)

add_mlir_dialect_library(MLIRLLVMIR
  IR/FunctionCallUtils.cpp
  IR/LLVMDialect.cpp
  IR/LLVMTypes.cpp
  IR/LLVMTypeSyntax.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  DEPENDS
  MLIRLLVMOpsIncGen
  MLIRLLVMOpsInterfacesIncGen
  MLIROpenMPOpsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  AsmParser
  BitReader
  BitWriter
  Core

  LINK_LIBS PUBLIC
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRDataLayoutInterfaces
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
  )

add_mlir_dialect_library(MLIRNVVMIR
  IR/NVVMDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  DEPENDS
  MLIRNVVMOpsIncGen
  MLIRNVVMConversionsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  AsmParser
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMIR
  MLIRSideEffectInterfaces
  )

add_mlir_dialect_library(MLIRROCDLIR
  IR/ROCDLDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  DEPENDS
  MLIRROCDLOpsIncGen
  MLIRROCDLConversionsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  AsmParser
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRVectorToLLVM
  )
