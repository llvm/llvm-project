add_llvm_component_group(Connex)

set(LLVM_TARGET_DEFINITIONS Connex.td)

tablegen(LLVM ConnexGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM ConnexGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM ConnexGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM ConnexGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM ConnexGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM ConnexGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM ConnexGenCallingConv.inc -gen-callingconv)
tablegen(LLVM ConnexGenSubtargetInfo.inc -gen-subtarget)
add_public_tablegen_target(ConnexCommonTableGen)

add_llvm_target(ConnexCodeGen
  ConnexAsmPrinter.cpp
  ConnexFrameLowering.cpp
  ConnexHazardRecognizer.cpp
  ConnexInstrInfo.cpp
  ConnexISelDAGToDAG.cpp
  ConnexISelLowering.cpp
  ConnexMCInstLower.cpp
  ConnexRegisterInfo.cpp
  ConnexSelectionDAGInfo.cpp
  ConnexSubtarget.cpp
  ConnexTargetMachine.cpp

  # similar to the ones of the Mips back end
  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  Core
  MC
  ConnexDesc
  ConnexInfo
  IPO
  Scalar
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT
  Connex
  )

# add_subdirectory(AsmParser)
# add_subdirectory(InstPrinter)
# add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
