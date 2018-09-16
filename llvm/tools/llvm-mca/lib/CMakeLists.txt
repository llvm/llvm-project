include_directories(${LLVM_MCA_SOURCE_DIR}/include)

add_library(LLVMMCA
  STATIC
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
  )

llvm_update_compile_flags(LLVMMCA)
llvm_map_components_to_libnames(libs
  CodeGen
  MC
  Support
  )

target_link_libraries(LLVMMCA ${libs})
set_target_properties(LLVMMCA PROPERTIES FOLDER "Libraries")
