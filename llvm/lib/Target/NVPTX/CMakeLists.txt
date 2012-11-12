set(LLVM_TARGET_DEFINITIONS NVPTX.td)


tablegen(LLVM NVPTXGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM NVPTXGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM NVPTXGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM NVPTXGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM NVPTXGenSubtargetInfo.inc -gen-subtarget)
add_public_tablegen_target(NVPTXCommonTableGen)

set(NVPTXCodeGen_sources
  NVPTXFrameLowering.cpp
  NVPTXInstrInfo.cpp
  NVPTXISelDAGToDAG.cpp
  NVPTXISelLowering.cpp
  NVPTXRegisterInfo.cpp
  NVPTXSubtarget.cpp
  NVPTXTargetMachine.cpp
  NVPTXSplitBBatBar.cpp
  NVPTXLowerAggrCopies.cpp
  NVPTXutil.cpp
  NVPTXAllocaHoisting.cpp
  NVPTXAsmPrinter.cpp
  NVPTXUtilities.cpp
  VectorElementize.cpp
  )

add_llvm_target(NVPTXCodeGen ${NVPTXCodeGen_sources})

add_dependencies(LLVMNVPTXCodeGen intrinsics_gen)

add_subdirectory(TargetInfo)
add_subdirectory(InstPrinter)
add_subdirectory(MCTargetDesc)
