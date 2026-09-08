add_subdirectory(Transforms)

set(LLVM_OPTIONAL_SOURCES
  IR/LLVMOps.cpp
  IR/ROCDLOps.cpp
  )

set(LLVM_TARGET_DEFINITIONS IR/LLVMOps.cpp)
foreach(index RANGE 7)
  set(LLVM_SHARDED_SRC LLVMOps.${index}.cpp)
  list(APPEND LLVM_SHARDED_SRCS ${LLVM_SHARDED_SRC})
  tablegen(MLIR_SRC_SHARDER ${LLVM_SHARDED_SRC}
    -op-shard-index=${index})
endforeach()
add_public_tablegen_target(MLIRLLVMOpsShardGen)

add_mlir_dialect_library(MLIRLLVMDialect
  IR/FunctionCallUtils.cpp
  IR/LLVMAttrs.cpp
  IR/LLVMDialect.cpp
  IR/LLVMInterfaces.cpp
  IR/LLVMMemorySlot.cpp
  IR/LLVMTypes.cpp
  IR/LLVMTypeSyntax.cpp
  IR/LLVMDialectBytecode.cpp
  ${LLVM_SHARDED_SRCS}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  DEPENDS
  MLIRLLVMDialectBytecodeIncGen
  MLIRLLVMInterfacesIncGen
  MLIRLLVMIntrinsicOpsIncGen
  MLIRLLVMAllOpsIncGen
  MLIRLLVMOpsIncGen
  MLIRLLVMOpsShardGen
  MLIRLLVMTypesIncGen
  intrinsics_gen
  MLIRLLVMConversionsIncGen
  MLIRLLVMIntrinsicConversionsIncGen

  LINK_COMPONENTS
  BinaryFormat
  Core

  HEADER_LIBS
  # LLVMAttrs.cpp includes mlir/Dialect/Ptr/IR/PtrEnums.h.
  MLIRPtrDialect
  # LLVMDialect.cpp includes mlir/Transforms/InliningUtils.h.
  MLIRTransformUtils

  LINK_LIBS PUBLIC
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRDataLayoutInterfaces
  MLIRFunctionInterfaces
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRMemorySlotInterfaces
  MLIRPtrMemorySpaceInterfaces
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
  MLIRNVVMOpsIncGen
  MLIRNVVMConversionsIncGen
  MLIRBasicPtxBuilderInterfaceIncGen
  MLIRNVVMRequiresSMTraitsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  AsmParser
  Core
  TargetParser

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRGPUDialect
  MLIROpenMPDialect
  MLIRSideEffectInterfaces
  MLIRInferIntRangeInterface
  )

set(LLVM_TARGET_DEFINITIONS IR/ROCDLOps.cpp)
foreach(index RANGE 7)
  set(ROCDL_SHARDED_SRC ROCDLOps.${index}.cpp)
  list(APPEND ROCDL_SHARDED_SRCS ${ROCDL_SHARDED_SRC})
  tablegen(MLIR_SRC_SHARDER ${ROCDL_SHARDED_SRC}
    -op-shard-index=${index})
endforeach()
add_public_tablegen_target(MLIRROCDLOpsShardGen)

add_mlir_dialect_library(MLIRROCDLDialect
  IR/ROCDLDialect.cpp
  ${ROCDL_SHARDED_SRCS}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  DEPENDS
  MLIRROCDLOpsIncGen
  MLIRROCDLOpsShardGen
  MLIRROCDLConversionsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  AsmParser
  Core

  # ROCDLDialect.cpp includes GPU CompilationInterfaces and InliningUtils.
  HEADER_LIBS
  MLIRGPUDialect
  MLIRTransformUtils

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
  MLIRVCIXOpsIncGen
  MLIRVCIXConversionsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  AsmParser
  Core

  # VCIXDialect.cpp includes the generated GPU CompilationInterfaces.
  HEADER_LIBS
  MLIRGPUDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSideEffectInterfaces
  )

add_mlir_dialect_library(MLIRXeVMDialect
  IR/XeVMDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  DEPENDS
  MLIRXeVMOpsIncGen
  MLIRXeVMConversionsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  AsmParser
  Core
  TargetParser

  # XeVMDialect.cpp includes the generated GPU CompilationInterfaces.
  HEADER_LIBS
  MLIRGPUDialect

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRLLVMDialect
  MLIROpenMPDialect
  MLIRSideEffectInterfaces
)
