set(LLVM_TARGET_DEFINITIONS NVPTX.td)

tablegen(LLVM NVPTXGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM NVPTXGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM NVPTXGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM NVPTXGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM NVPTXGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(NVPTXCommonTableGen)

set(NVPTXCodeGen_sources
  NVPTXAllocaHoisting.cpp
  NVPTXAsmPrinter.cpp
  NVPTXAssignValidGlobalNames.cpp
  NVPTXFrameLowering.cpp
  NVPTXGenericToNVVM.cpp
  NVPTXISelDAGToDAG.cpp
  NVPTXISelLowering.cpp
  NVPTXImageOptimizer.cpp
  NVPTXInstrInfo.cpp
  NVPTXLowerAggrCopies.cpp
  NVPTXLowerArgs.cpp
  NVPTXLowerAlloca.cpp
  NVPTXPeephole.cpp
  NVPTXMCExpr.cpp
  NVPTXPrologEpilogPass.cpp
  NVPTXRegisterInfo.cpp
  NVPTXReplaceImageHandles.cpp
  NVPTXSubtarget.cpp
  NVPTXTargetMachine.cpp
  NVPTXTargetTransformInfo.cpp
  NVPTXUtilities.cpp
  NVVMIntrRange.cpp
  NVVMReflect.cpp
  )

add_llvm_target(NVPTXCodeGen ${NVPTXCodeGen_sources})

add_subdirectory(InstPrinter)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
