add_llvm_component_library(LLVMRuntimeDyld
  JITSymbol.cpp
  RTDyldMemoryManager.cpp
  RuntimeDyld.cpp
  RuntimeDyldChecker.cpp
  RuntimeDyldCOFF.cpp
  RuntimeDyldELF.cpp
  RuntimeDyldMachO.cpp
  Targets/RuntimeDyldELFMips.cpp

  DEPENDS
  intrinsics_gen


  LINK_COMPONENTS
  Core
  MC
  Object
  Support
  TargetParser
  )

if ("${LLVM_TARGETS_TO_BUILD}" MATCHES "Hexagon")
  add_definitions(-DHEXAGON_TARGET_ENABLED)
  target_link_libraries(LLVMRuntimeDyld PRIVATE LLVMHexagonCodeGen)
endif()
