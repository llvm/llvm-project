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

# Static relocation helper functions (or32le, getBits, etc.) conflict.
set_source_files_properties(
  RuntimeDyldELF.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
