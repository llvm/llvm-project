add_llvm_component_group(M68k)

set(LLVM_TARGET_DEFINITIONS M68k.td)

tablegen(LLVM M68kGenGlobalISel.inc       -gen-global-isel)
tablegen(LLVM M68kGenRegisterInfo.inc     -gen-register-info)
tablegen(LLVM M68kGenRegisterBank.inc     -gen-register-bank)
tablegen(LLVM M68kGenInstrInfo.inc        -gen-instr-info)
tablegen(LLVM M68kGenSubtargetInfo.inc    -gen-subtarget)
tablegen(LLVM M68kGenMCCodeBeads.inc      -gen-code-beads)
tablegen(LLVM M68kGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM M68kGenDAGISel.inc          -gen-dag-isel)
tablegen(LLVM M68kGenCallingConv.inc      -gen-callingconv)
tablegen(LLVM M68kGenAsmWriter.inc        -gen-asm-writer)
tablegen(LLVM M68kGenAsmMatcher.inc       -gen-asm-matcher)

add_public_tablegen_target(M68kCommonTableGen)

add_llvm_target(M68kCodeGen
  GlSel/M68kCallLowering.cpp
  GlSel/M68kInstructionSelector.cpp
  GlSel/M68kLegalizerInfo.cpp
  GlSel/M68kRegisterBankInfo.cpp
  M68kAsmPrinter.cpp
  M68kCollapseMOVEMPass.cpp
  M68kExpandPseudo.cpp
  M68kFrameLowering.cpp
  M68kInstrInfo.cpp
  M68kISelLowering.cpp
  M68kISelDAGToDAG.cpp
  M68kMachineFunction.cpp
  M68kMCInstLower.cpp
  M68kRegisterInfo.cpp
  M68kSubtarget.cpp
  M68kTargetMachine.cpp
  M68kTargetObjectFile.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  Core
  GlobalISel
  MC
  SelectionDAG
  Support
  Target
  M68kDesc
  M68kInfo

  ADD_TO_COMPONENT
  M68k
)

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
