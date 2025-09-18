if (UNIX AND "${CMAKE_SYSTEM_NAME}" MATCHES "AIX")
  set(additional_libs bsd)
endif()

add_llvm_component_library(LLVMCAS
  ActionCache.cpp
  ActionCaches.cpp
  BuiltinCAS.cpp
  InMemoryCAS.cpp
  MappedFileRegionArena.cpp
  ObjectStore.cpp
  OnDiskCommon.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/CAS

  LINK_LIBS
  ${additional_libs}

  LINK_COMPONENTS
  Support
)
