set(LLVM_TARGET_DEFINITIONS Toy64.td)

include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Generate the C++ register descriptions from Toy64RegisterInfo.td.
tablegen(LLVM Toy64GenRegisterInfo.inc -gen-register-info)
tablegen(LLVM Toy64GenInstrInfo.inc -gen-instr-info)
tablegen(LLVM Toy64GenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM Toy64GenDAGISel.inc -gen-dag-isel)

# Group all common generated TableGen files behind one build target.
add_public_tablegen_target(Toy64CommonTableGen)

add_llvm_component_group(Toy64)

add_llvm_component_library(LLVMToy64CodeGen
  Toy64AsmPrinter.cpp
  Toy64FrameLowering.cpp
  Toy64InstrInfo.cpp
  Toy64ISelDAGToDAG.cpp
  Toy64ISelLowering.cpp
  Toy64RegisterInfo.cpp
  Toy64Subtarget.cpp
  Toy64TargetMachine.cpp

  DEPENDS
  Toy64CommonTableGen

  LINK_COMPONENTS
  CodeGen
  Core
  MC
  SelectionDAG
  Support
  Target
  Toy64Desc
  Toy64Info

  ADD_TO_COMPONENT
  Toy64
)

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)