set(LLVM_TARGET_DEFINITIONS XCore.td)

tablegen(XCoreGenRegisterInfo.h.inc -gen-register-desc-header)
tablegen(XCoreGenRegisterNames.inc -gen-register-enums)
tablegen(XCoreGenRegisterInfo.inc -gen-register-desc)
tablegen(XCoreGenInstrNames.inc -gen-instr-enums)
tablegen(XCoreGenInstrInfo.inc -gen-instr-desc)
tablegen(XCoreGenAsmWriter.inc -gen-asm-writer)
tablegen(XCoreGenDAGISel.inc -gen-dag-isel)
tablegen(XCoreGenCallingConv.inc -gen-callingconv)
tablegen(XCoreGenSubtarget.inc -gen-subtarget)

add_llvm_target(XCore
  XCoreAsmPrinter.cpp
  XCoreFrameInfo.cpp
  XCoreInstrInfo.cpp
  XCoreISelDAGToDAG.cpp
  XCoreISelLowering.cpp
  XCoreRegisterInfo.cpp
  XCoreSubtarget.cpp
  XCoreTargetAsmInfo.cpp
  XCoreTargetMachine.cpp
  )
