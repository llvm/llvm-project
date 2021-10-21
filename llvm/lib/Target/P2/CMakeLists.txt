add_llvm_component_group(P2)

set(LLVM_TARGET_DEFINITIONS P2.td)

tablegen(LLVM P2GenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM P2GenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM P2GenCallingConv.inc -gen-callingconv)
tablegen(LLVM P2GenCodeEmitter.inc -gen-emitter)
tablegen(LLVM P2GenDAGISel.inc -gen-dag-isel)
tablegen(LLVM P2GenInstrInfo.inc -gen-instr-info)
tablegen(LLVM P2GenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM P2GenRegisterInfo.inc -gen-register-info)
tablegen(LLVM P2GenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM P2GenDisassemblerTables.inc -gen-disassembler)

#tablegen(LLVM P2GenMCPseudoLowering.inc -gen-pseudo-lowering)
add_public_tablegen_target(P2CommonTableGen)

add_llvm_target(P2CodeGen
    P2AsmPrinter.cpp
    P2ExpandPseudos.cpp
    P2FrameLowering.cpp
    P2InstrInfo.cpp
    P2ISelLowering.cpp
    P2RegisterInfo.cpp
    P2InstrInfo.cpp
    P2RegisterInfo.cpp
    P2TargetObjectFile.cpp
    P2Subtarget.cpp
    P2TargetMachine.cpp
    P2FrameLowering.cpp
    P2MachineFunctionInfo.cpp
    P2MCInstLower.cpp
    P2ISelDAGToDAG.cpp

    DEPENDS
    intrinsics_gen

    LINK_COMPONENTS
    AsmPrinter
    CodeGen
    Core
    MC
    P2Desc
    P2Info
    SelectionDAG
    Support
    Target

    ADD_TO_COMPONENT
    P2
  )

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
add_subdirectory(AsmParser)
add_subdirectory(Disassembler)