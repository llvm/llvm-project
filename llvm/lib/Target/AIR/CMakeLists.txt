add_llvm_component_group(AIR)

set(LLVM_TARGET_DEFINITIONS AIR.td)

tablegen(LLVM AIRGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM AIRGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM AIRGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM AIRGenAIRIntrinsicMappings.inc -gen-searchable-tables)

add_public_tablegen_target(AIRCommonTableGen)

add_llvm_target(AIRCodeGen
  AIRAliasAnalysis.cpp
  AIRSystemValues.cpp
  AIRAliasAnnotate.cpp
  AIRArgumentBuffer.cpp
  AIRAsyncEventToAlloca.cpp
  AIRBFloat16CastDecompose.cpp
  AIRBarrierRename.cpp
  AIRCrossBufferStoreSeparate.cpp
  AIRFromGenericGPU.cpp
  AIRDemoteF64.cpp
  AIREmulateF64.cpp
  AIRDeviceLoadsVolatile.cpp
  AIRInlineNonKernel.cpp
  AIRKernelParams.cpp
  LLVMToAIRIntrinsics.cpp
  AIRLowerAtomicRMW.cpp
  AIRLowerFNeg.cpp
  AIRLegalizeUnsupportedIR.cpp
  AIRNaNMinMax.cpp
  AIRNormalizeAllocas.cpp
  AIRPrepare.cpp
  AIRScalarArgsToConstantBuffers.cpp
  AIRScalarBufferPacking.cpp
  AIRScalarStoreGuard.cpp
  AIRSimdDivergenceGuard.cpp
  AIRSplitI64Shuffle.cpp
  AIRTGBarrierInsert.cpp
  AIRTGGlobalCoalesce.cpp
  AIRAsmPrinter.cpp
  AIRInstrInfo.cpp
  AIRRegisterInfo.cpp
  AIRSubtarget.cpp
  AIRTargetMachine.cpp

  LINK_COMPONENTS
  Analysis
  AsmParser
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  Linker
  MC
  Object
  Passes
  ScalarOpts
  AIRWriter
  AIRDesc
  AIRInfo
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  AIR
  )

add_subdirectory(AIRWriter)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
