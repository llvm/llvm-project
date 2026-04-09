# core/CMakeLists.txt
#
# Builds LLVMLX32CodeGen — the CodeGen layer of the LX32 backend.
#
# Source files are listed in build-dependency order: each file may only
# include headers from files listed earlier in this list.
#
#   LX32Subtarget.cpp     — TableGen-generated subtarget tables + constructor.
#                           No dependencies on other LX32 CodeGen sources.
#   LX32TargetMachine.cpp — Entry point; depends on LX32Subtarget.
#
#
# LINK_COMPONENTS declares which LLVM component libraries must be linked.
# Each component is resolved by the LLVM CMake infrastructure to the
# correct library (e.g., CodeGen → LLVMCodeGen.a / libLLVMCodeGen.so).
#
# Required components for the current skeleton:
#   CodeGen      — MachineFunction, TargetPassConfig, RA infrastructure
#   Core         — Type, Module, IRBuilder (needed by ISelLowering, Day 9)
#   MC           — MCAsmInfo, MCInstrInfo, MCRegisterInfo
#   SelectionDAG — SelectionDAG, DAGToDAGISel (needed by ISelDAGToDAG, Day 10)
#   Support      — ErrorHandling, Debug, raw_ostream
#   Target       — TargetMachine base class, TargetOptions

add_llvm_component_library(LLVMLX32CodeGen
  LX32AsmPrinter.cpp
  LX32FrameLowering.cpp
  LX32ISelDAGToDAG.cpp
  LX32InstrInfo.cpp
  LX32ISelLowering.cpp
  LX32RegisterInfo.cpp
  LX32Subtarget.cpp
  LX32TargetMachine.cpp

  LINK_COMPONENTS
    CodeGen
    Core
    MC
    SelectionDAG
    Support
    Target

  ADD_TO_COMPONENT
  LX32
)
