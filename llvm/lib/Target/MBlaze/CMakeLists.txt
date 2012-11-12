set(LLVM_TARGET_DEFINITIONS MBlaze.td)

tablegen(LLVM MBlazeGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM MBlazeGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM MBlazeGenCodeEmitter.inc -gen-emitter)
tablegen(LLVM MBlazeGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM MBlazeGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM MBlazeGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM MBlazeGenCallingConv.inc -gen-callingconv)
tablegen(LLVM MBlazeGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM MBlazeGenIntrinsics.inc -gen-tgt-intrinsic)
tablegen(LLVM MBlazeGenEDInfo.inc -gen-enhanced-disassembly-info)
add_public_tablegen_target(MBlazeCommonTableGen)

add_llvm_target(MBlazeCodeGen
  MBlazeDelaySlotFiller.cpp
  MBlazeInstrInfo.cpp
  MBlazeISelDAGToDAG.cpp
  MBlazeISelLowering.cpp
  MBlazeFrameLowering.cpp
  MBlazeMachineFunction.cpp
  MBlazeRegisterInfo.cpp
  MBlazeSubtarget.cpp
  MBlazeTargetMachine.cpp
  MBlazeTargetObjectFile.cpp
  MBlazeIntrinsicInfo.cpp
  MBlazeSelectionDAGInfo.cpp
  MBlazeAsmPrinter.cpp
  MBlazeMCInstLower.cpp
  )

add_dependencies(LLVMMBlazeCodeGen intrinsics_gen)

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(InstPrinter)
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
