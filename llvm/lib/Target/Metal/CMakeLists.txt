add_llvm_component_group(Metal)

set(LLVM_TARGET_DEFINITIONS Metal.td)

tablegen(LLVM MetalGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM MetalGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM MetalGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM MetalGenAIRIntrinsicMappings.inc -gen-searchable-tables)

add_public_tablegen_target(MetalCommonTableGen)

add_llvm_target(MetalCodeGen
  MetalAIRSystemValues.cpp
  MetalAliasAnnotate.cpp
  MetalAsyncEventToAlloca.cpp
  MetalBFloat16CastDecompose.cpp
  MetalBarrierRename.cpp
  MetalDeviceLoadsVolatile.cpp
  MetalInlineNonKernel.cpp
  MetalLLVMToAIRIntrinsics.cpp
  MetalLowerAtomicRMW.cpp
  MetalLowerFNeg.cpp
  MetalNaNMinMax.cpp
  MetalNormalizeAllocas.cpp
  MetalPrepare.cpp
  MetalScalarBufferPacking.cpp
  MetalScalarStoreGuard.cpp
  MetalSplitI64Shuffle.cpp
  MetalTGBarrierInsert.cpp
  MetalTGGlobalCoalesce.cpp
  MetalAsmPrinter.cpp
  MetalInstrInfo.cpp
  MetalRegisterInfo.cpp
  MetalSubtarget.cpp
  MetalTargetMachine.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  MC
  MetalAIRWriter
  MetalDesc
  MetalInfo
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  Metal
  )

add_subdirectory(AIRWriter)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
