set(LLDB_SYSTEM_LIBS)

list(APPEND LLDB_SYSTEM_LIBS ${system_libs})

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
  Connection.cpp
  ConstString.cpp
  CompletionRequest.cpp
  DataBufferHeap.cpp
  DataBufferLLVM.cpp
  DataEncoder.cpp
  DataExtractor.cpp
  Environment.cpp
  FileCollector.cpp
  Event.cpp
  FileSpec.cpp
  IOObject.cpp
  JSON.cpp
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
  Scalar.cpp
  SelectHelper.cpp
  SharingPtr.cpp
  State.cpp
  Status.cpp
  Stream.cpp
  StreamCallback.cpp
  StreamGDBRemote.cpp
  StreamString.cpp
  StringExtractor.cpp
  StringExtractorGDBRemote.cpp
  StringLexer.cpp
  StringList.cpp
  StructuredData.cpp
  TildeExpressionResolver.cpp
  Timer.cpp
  UserID.cpp
  UserIDResolver.cpp
  UriParser.cpp
  UUID.cpp
  VASprintf.cpp
  VMRange.cpp

  LINK_LIBS
    ${LLDB_SYSTEM_LIBS}
    # lldbUtility does not depend on other LLDB libraries

  LINK_COMPONENTS
    BinaryFormat
    Support
  )
