add_subdirectory(Transforms)

set(LLVM_OPTIONAL_SOURCES
  IR/ROCDLOps.cpp
  )

add_mlir_dialect_library(MLIRLLVMDialect
  IR/FunctionCallUtils.cpp
  IR/LLVMAttrs.cpp
  IR/LLVMDialect.cpp
  IR/LLVMInterfaces.cpp
  IR/LLVMTypes.cpp
  IR/LLVMTypeSyntax.cpp
  IR/LLVMDialectBytecode.cpp
  IR/LLVMAggregateOps.cpp
  IR/LLVMArithOps.cpp
  IR/LLVMCallOps.cpp
  IR/LLVMCodeGenIntrinsicOps.cpp
  IR/LLVMConstantOps.cpp
  IR/LLVMConstrainedIntrinsicOps.cpp
  IR/LLVMConversionOps.cpp
  IR/LLVMCoroIntrinsicOps.cpp
  IR/LLVMDebugIntrinsicOps.cpp
  IR/LLVMFloatIntrinsicOps.cpp
  IR/LLVMGeneralIntrinsicOps.cpp
  IR/LLVMIntegerIntrinsicOps.cpp
  IR/LLVMMemoryIntrinsicOps.cpp
  IR/LLVMMemoryOps.cpp
  IR/LLVMModuleOps.cpp
  IR/LLVMTerminatorOps.cpp
  IR/LLVMVarArgOps.cpp
  IR/LLVMVectorIntrinsicOps.cpp
  IR/LLVMVectorOps.cpp
  IR/LLVMVPIntrinsicOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  PRECOMPILE_HEADERS
  [["mlir/Dialect/LLVMIR/pch.h"]]

  DEPENDS
  MLIRSCFToControlFlowDialectInterfaceIncGen
  MLIRLLVMDialectBytecodeIncGen
  MLIRLLVMInterfacesIncGen
  MLIRLLVMAllOpsIncGen
  MLIRLLVMOpsIncGen
  MLIRLLVMTranslationDialectInterfaceIncGen
  MLIRLLVMTypesIncGen
  MLIROpenMPOpsIncGen
  intrinsics_gen
  MLIRLLVMConversionsIncGen
  MLIRLLVMIntrinsicConversionsIncGen

  LINK_COMPONENTS
  BinaryFormat
  Core

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
  IR/NVVMAttributes.cpp
  IR/NVVMDialect.cpp
  IR/BasicPtxBuilderInterface.cpp
  IR/NVVMRequiresSMTraits.cpp
  IR/NVVMArithOps.cpp
  IR/NVVMAsyncCopyOps.cpp
  IR/NVVMConvertOps.cpp
  IR/NVVMDataMovementOps.cpp
  IR/NVVMIndexSRegOps.cpp
  IR/NVVMMBarrierOps.cpp
  IR/NVVMMiscOps.cpp
  IR/NVVMMiscSRegOps.cpp
  IR/NVVMMmaOps.cpp
  IR/NVVMSyncOps.cpp
  IR/NVVMTcgen05Ops.cpp
  IR/NVVMWgmmaOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  # Cache the operation declarations shared by all source files. Keep this
  # specialized PCH local rather than making downstream libraries reuse it.
  DISABLE_PCH_REUSE
  PRECOMPILE_HEADERS
  [["mlir/Dialect/LLVMIR/pch.h"]]
  [["mlir/Dialect/LLVMIR/NVVMDialect.h"]]

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
  TargetParser

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRGPUDialect
  MLIRSideEffectInterfaces
  MLIRInferIntRangeInterface
  )

set(LLVM_TARGET_DEFINITIONS IR/ROCDLOps.cpp)
foreach(index RANGE 7)
  set(ROCDL_SHARDED_SRC ROCDLOps.${index}.cpp)
  list(APPEND ROCDL_SHARDED_SRCS ${ROCDL_SHARDED_SRC})
  mlir_src_sharder(${ROCDL_SHARDED_SRC}
    -op-shard-index=${index})
endforeach()
add_public_tablegen_target(MLIRROCDLOpsShardGen)

add_mlir_dialect_library(MLIRROCDLDialect
  IR/ROCDLDialect.cpp
  ${ROCDL_SHARDED_SRCS}
  IR/ROCDLTargetInfo.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  # Cache the operation declarations shared by all source shards. Keep this
  # specialized PCH local rather than making downstream libraries reuse it.
  DISABLE_PCH_REUSE
  PRECOMPILE_HEADERS
  [["mlir/Dialect/LLVMIR/pch.h"]]
  [["mlir/Dialect/LLVMIR/ROCDLDialect.h"]]

  DEPENDS
  MLIRGPUCompilationAttrInterfacesIncGen
  MLIRROCDLOpsIncGen
  MLIRROCDLOpsShardGen
  MLIRROCDLConversionsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  AsmParser
  Core
  TargetParser

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
  MLIRGPUCompilationAttrInterfacesIncGen
  MLIRXeVMOpsIncGen
  MLIRXeVMConversionsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  AsmParser
  Core
  TargetParser

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRLLVMDialect
  MLIRSideEffectInterfaces
)
