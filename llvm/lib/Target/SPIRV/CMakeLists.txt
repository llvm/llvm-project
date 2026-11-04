add_llvm_component_group(SPIRV)

set(LLVM_TARGET_DEFINITIONS SPIRV.td)

llvm_tablegen(SPIRVGenAsmWriter.inc -gen-asm-writer)
llvm_tablegen(SPIRVGenGlobalISel.inc -gen-global-isel)
llvm_tablegen(SPIRVGenInstrInfo.inc -gen-instr-info)
llvm_tablegen(SPIRVGenMCCodeEmitter.inc -gen-emitter)
llvm_tablegen(SPIRVGenRegisterBank.inc -gen-register-bank)
llvm_tablegen(SPIRVGenRegisterInfo.inc -gen-register-info)
llvm_tablegen(SPIRVGenSubtargetInfo.inc -gen-subtarget)
llvm_tablegen(SPIRVGenTables.inc -gen-searchable-tables)
llvm_tablegen(SPIRVGenPreLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="SPIRVPreLegalizerCombiner")

add_public_tablegen_target(SPIRVCommonTableGen)

add_llvm_target(SPIRVCodeGen
  SPIRVAPI.cpp
  SPIRVAsmPrinter.cpp
  SPIRVBuiltins.cpp
  SPIRVCallLowering.cpp
  SPIRVCodeGenPassBuilder.cpp
  SPIRVInlineAsmLowering.cpp
  SPIRVCommandLine.cpp
  SPIRVCtorDtorLowering.cpp
  SPIRVEmitIntrinsics.cpp
  SPIRVGlobalRegistry.cpp
  SPIRVInstrInfo.cpp
  SPIRVInstructionSelector.cpp
  SPIRVLegalizeResourceBinding.cpp
  SPIRVLegalizeZeroSizeArrays.cpp
  SPIRVLegalizePointerCast.cpp
  SPIRVMergeRegionExitTargets.cpp
  SPIRVISelLowering.cpp
  SPIRVLegalizerInfo.cpp
  SPIRVMCInstLower.cpp
  SPIRVMetadata.cpp
  SPIRVModuleAnalysis.cpp
  SPIRVStructurizer.cpp
  SPIRVCombinerHelper.cpp
  SPIRVPreLegalizer.cpp
  SPIRVPreLegalizerCombiner.cpp
  SPIRVPostLegalizer.cpp
  SPIRVPrepareFunctions.cpp
  SPIRVPrepareGlobals.cpp
  SPIRVRegisterBankInfo.cpp
  SPIRVRegisterInfo.cpp
  SPIRVRegularizer.cpp
  SPIRVFinalizeShaderLinkage.cpp
  SPIRVSubtarget.cpp
  SPIRVTargetMachine.cpp
  SPIRVTargetTransformInfo.cpp
  SPIRVTypeInst.cpp
  SPIRVUtils.cpp
  SPIRVNonSemanticDebugHandler.cpp
  SPIRVAuxDataHandler.cpp
  SPIRVCBufferAccess.cpp
  SPIRVPushConstantAccess.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  Demangle
  GlobalISel
  FrontendHLSL
  IPO
  MC
  Passes
  SPIRVAnalysis
  SPIRVDesc
  SPIRVInfo
  ScalarOpts
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  SPIRV
  )

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(Analysis)
