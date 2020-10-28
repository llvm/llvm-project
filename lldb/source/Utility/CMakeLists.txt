set(LLDB_SYSTEM_LIBS)

if (APPLE)
  list(APPEND LLDB_SYSTEM_LIBS
       ${FOUNDATION_LIBRARY}
       ${CORE_FOUNDATION_LIBRARY}
       ${CORE_SERVICES_LIBRARY}
       ${SECURITY_LIBRARY}
       ${DEBUG_SYMBOLS_LIBRARY})
endif()

if(NOT PURE_WINDOWS)
  list(APPEND LLDB_SYSTEM_LIBS ${CMAKE_THREAD_LIBS_INIT})
endif()

list(APPEND LLDB_SYSTEM_LIBS ${CMAKE_DL_LIBS})

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
  list(APPEND LLDB_SYSTEM_LIBS ws2_32 rpcrt4)
endif ()

if (NOT HAVE_CXX_ATOMICS64_WITHOUT_LIB )
    list(APPEND LLDB_SYSTEM_LIBS atomic)
endif()

add_lldb_library(lldbUtility
  ArchSpec.cpp
  Args.cpp
  Baton.cpp
  Broadcaster.cpp
  CompletionRequest.cpp
  Connection.cpp
  ConstString.cpp
  DataBufferHeap.cpp
  DataBufferLLVM.cpp
  DataEncoder.cpp
  DataExtractor.cpp
  Environment.cpp
  Event.cpp
  FileSpec.cpp
  GDBRemote.cpp
  IOObject.cpp
  LLDBAssert.cpp
  Listener.cpp
  Log.cpp
  Logging.cpp
  NameMatches.cpp
  ProcessInfo.cpp
  RegisterValue.cpp
  RegularExpression.cpp
  Reproducer.cpp
  ReproducerInstrumentation.cpp
  ReproducerProvider.cpp
  Scalar.cpp
  SelectHelper.cpp
  State.cpp
  Status.cpp
  Stream.cpp
  StreamCallback.cpp
  StreamString.cpp
  StringExtractor.cpp
  StringExtractorGDBRemote.cpp
  StringLexer.cpp
  StringList.cpp
  StructuredData.cpp
  TildeExpressionResolver.cpp
  Timer.cpp
  UnimplementedError.cpp
  UUID.cpp
  UriParser.cpp
  UserID.cpp
  UserIDResolver.cpp
  VASprintf.cpp
  VMRange.cpp
  XcodeSDK.cpp

  LINK_LIBS
    ${LLDB_SYSTEM_LIBS}
    # lldbUtility does not depend on other LLDB libraries

  LINK_COMPONENTS
    BinaryFormat
    Support
  )
