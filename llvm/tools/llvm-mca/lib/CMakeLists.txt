include_directories(${LLVM_MCA_SOURCE_DIR}/include)

add_llvm_library(LLVMMCA
  Context.cpp
  HWEventListener.cpp
  HardwareUnits/HardwareUnit.cpp
  HardwareUnits/LSUnit.cpp
  HardwareUnits/RegisterFile.cpp
  HardwareUnits/ResourceManager.cpp
  HardwareUnits/RetireControlUnit.cpp
  HardwareUnits/Scheduler.cpp
  InstrBuilder.cpp
  Instruction.cpp
  Pipeline.cpp
  Stages/DispatchStage.cpp
  Stages/ExecuteStage.cpp
  Stages/FetchStage.cpp
  Stages/InstructionTables.cpp
  Stages/RetireStage.cpp
  Stages/Stage.cpp
  Support.cpp

  STATIC

  LINK_COMPONENTS
  CodeGen
  MC
  Support
  )
