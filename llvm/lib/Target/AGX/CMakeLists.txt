add_llvm_component_group(AGX)

set(LLVM_TARGET_DEFINITIONS AGX.td)

tablegen(LLVM AGXGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM AGXGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM AGXGenCallingConv.inc -gen-callingconv)
tablegen(LLVM AGXGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM AGXGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM AGXGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM AGXGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM AGXGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM AGXGenSDNodeInfo.inc -gen-sd-node-info)
tablegen(LLVM AGXGenSearchableTables.inc -gen-searchable-tables)
tablegen(LLVM AGXGenSubtargetInfo.inc -gen-subtarget)

# The AIR target's mapping between LLVM intrinsics and AIR functions, which
# agx-lower-air inverts. Generated here from the same file, so that the AGX
# target does not depend on the AIR target being built.
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/../AIR/AIRIntrinsicMappings.td)
tablegen(LLVM AGXGenAIRFunctions.inc -gen-searchable-tables)
set(LLVM_TARGET_DEFINITIONS AGX.td)

add_public_tablegen_target(AGXCommonTableGen)

add_llvm_target(AGXCodeGen
  AGXAnnotateControlFlow.cpp
  AGXAsmPrinter.cpp
  AGXCodeGenPrepare.cpp
  AGXMathExpansion.cpp
  AGXMemCluster.cpp
  AGXUniformConstants.cpp
  AGXLowerRecursion.cpp
  AGXFoldOperands.cpp
  AGXHoistFromElseFlow.cpp
  AGXFrameLowering.cpp
  AGXHoistGuardedLoads.cpp
  AGXISelDAGToDAG.cpp
  AGXISelLowering.cpp
  AGXInsertWaits.cpp
  AGXInstrInfo.cpp
  AGXLowerControlFlow.cpp
  AGXCheckedAccesses.cpp
  AGXLowerAIR.cpp
  AGXAliasAnalysis.cpp
  AGXLowerGenericAddresses.cpp
  AGXMachineFunctionInfo.cpp
  AGXMCInstLower.cpp
  AGXPromoteKernelArgs.cpp
  AGXRegisterInfo.cpp
  AGXSelectionDAGInfo.cpp
  AGXExpandWideAtomics.cpp
  AGXSinkAddresses.cpp
  AGXThreadgroupAddresses.cpp
  AGXSoftF64.cpp
  AGXStrictFP.cpp
  AGXSubtarget.cpp
  AGXTargetMachine.cpp
  AGXTargetTransformInfo.cpp

  LINK_COMPONENTS
  AGXDesc
  AGXInfo
  AGXUtils
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  IPO
  InstCombine
  MC
  Passes
  Scalar
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  Vectorize

  ADD_TO_COMPONENT
  AGX
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(Utils)
