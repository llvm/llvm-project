add_subdirectory(Transforms)

add_mlir_dialect_library(MLIRLLVMDialect
  IR/FunctionCallUtils.cpp
  IR/LLVMAttrs.cpp
  IR/LLVMDialect.cpp
  IR/LLVMInterfaces.cpp
  IR/LLVMMemorySlot.cpp
  IR/LLVMTypes.cpp
  IR/LLVMTypeSyntax.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  DEPENDS
  MLIRLLVMOpsIncGen
  MLIRLLVMTypesIncGen
  MLIRLLVMIntrinsicOpsIncGen
  MLIRLLVMInterfacesIncGen
  MLIROpenMPOpsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  AsmParser
  BinaryFormat
  BitReader
  BitWriter
  Core

  LINK_LIBS PUBLIC
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRDataLayoutInterfaces
  MLIRFunctionInterfaces
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRMemorySlotInterfaces
  MLIRSideEffectInterfaces
  MLIRSupport
  )

add_mlir_dialect_library(MLIRNVVMDialect
  IR/NVVMDialect.cpp
  IR/BasicPtxBuilderInterface.cpp
  IR/NVVMRequiresSMTraits.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  DEPENDS
  MLIRGPUCompilationAttrInterfacesIncGen
  MLIRNVVMOpsIncGen
  MLIRNVVMConversionsIncGen
  MLIRBasicPtxBuilderInterfaceIncGen
  MLIRNVVMRequiresSMTraitsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  AsmParser
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRGPUDialect
  MLIRSideEffectInterfaces
  MLIRInferIntRangeInterface
  )

add_mlir_dialect_library(MLIRROCDLDialect
  IR/ROCDLDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  DEPENDS
  MLIRGPUCompilationAttrInterfacesIncGen
  MLIRROCDLOpsIncGen
  MLIRROCDLConversionsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  AsmParser
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSideEffectInterfaces
  )

add_mlir_dialect_library(MLIRVCIXDialect
  IR/VCIXDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  DEPENDS
  MLIRGPUCompilationAttrInterfacesIncGen
  MLIRVCIXOpsIncGen
  MLIRVCIXConversionsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  AsmParser
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSideEffectInterfaces
  )
