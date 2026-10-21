add_llvm_component_group(Genpay)
set(LLVM_TARGET_DEFINITIONS Genpay.td)

tablegen(LLVM GenpayGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM GenpayGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM GenpayGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM GenpayGenRegisterInfo.inc -gen-register-info)

add_public_tablegen_target(GenpayCommonTableGen)

add_llvm_target(GenpayCodeGen
  GenpayAsmPrinter.cpp
  GenpayFrameLowering.cpp
  GenpayISelDAGToDAG.cpp
  GenpayISelLowering.cpp
  GenpayInstrInfo.cpp
  GenpayRegisterInfo.cpp
  GenpaySubtarget.cpp
  GenpayTargetMachine.cpp

  DEPENDS
  GenpayCommonTableGen
  intrinsics_gen

  LINK_COMPONENTS
  GenpayDesc
  GenpayInfo
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  MC
  SelectionDAG
  Support
  Target

  ADD_TO_COMPONENT
  Genpay
  )

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
