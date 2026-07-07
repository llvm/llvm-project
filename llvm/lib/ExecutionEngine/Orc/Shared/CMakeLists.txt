if("${CMAKE_SYSTEM_NAME}" MATCHES "QNX")
  # On QNX the socket APIs (shutdown, used by FDSimpleRemoteEPCTransport in
  # SimpleRemoteEPCUtils.cpp) live in libsocket rather than libc. On SunOS and
  # Haiku the equivalent library is already pulled in transitively via Support.
  set(ORC_SHARED_EXTRA_LIBS socket)
endif()

add_llvm_component_library(LLVMOrcShared
  AllocationActions.cpp
  MachOObjectFormat.cpp
  ObjectFormats.cpp
  OrcError.cpp
  OrcRTBridge.cpp
  SimpleRemoteEPCUtils.cpp
  SymbolStringPool.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ExecutionEngine/Orc/Shared

  DEPENDS
  intrinsics_gen

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}
  ${ORC_SHARED_EXTRA_LIBS}

  LINK_COMPONENTS
  Support
  )
