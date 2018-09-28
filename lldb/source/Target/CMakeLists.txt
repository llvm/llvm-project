include_directories(../Plugins/Process/Utility)

set(REFLECTION_LIB "")

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(REFLECTION_LIB "swiftReflection-linux-x86_64")
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    # libswiftReflection is placed in a subdirectory which depends
    # on the host we're currently running on.
    # FIXME: There must be a better way to get at libswiftReflection.a
    string(REPLACE "-" ";" TRIPLE_LIST "${LLVM_HOST_TRIPLE}")
    list(GET TRIPLE_LIST 0 ARCH)
    list(GET TRIPLE_LIST 2 HOST_OS_VERSION)
    string(REGEX MATCHALL "[a-z]+" HOST_OS "${HOST_OS_VERSION}")
    set(REFLECTION_LIB "${REFLECTION_LIB}-${HOST_OS}-${ARCH}")
    message("Building for non-linux host, setting reflection lib to ${REFLECTION_LIB}")
endif()


add_lldb_library(lldbTarget
  ABI.cpp
  CPPLanguageRuntime.cpp
  ExecutionContext.cpp
  FileAction.cpp
  JITLoader.cpp
  JITLoaderList.cpp
  InstrumentationRuntime.cpp
  InstrumentationRuntimeStopInfo.cpp
  Language.cpp
  LanguageRuntime.cpp
  Memory.cpp
  MemoryHistory.cpp
  ModuleCache.cpp
  ObjCLanguageRuntime.cpp
  OperatingSystem.cpp
  PathMappingList.cpp
  Platform.cpp
  Process.cpp
  ProcessInfo.cpp
  ProcessLaunchInfo.cpp
  Queue.cpp
  QueueItem.cpp
  QueueList.cpp
  RegisterContext.cpp
  RegisterNumber.cpp
  SectionLoadHistory.cpp
  SectionLoadList.cpp
  StackFrame.cpp
  StackFrameList.cpp
  StackID.cpp
  StopInfo.cpp
  StructuredDataPlugin.cpp
  SwiftLanguageRuntime.cpp
  SystemRuntime.cpp
  Target.cpp
  TargetList.cpp
  Thread.cpp
  ThreadCollection.cpp
  ThreadList.cpp
  ThreadPlan.cpp
  ThreadPlanBase.cpp
  ThreadPlanCallFunction.cpp
  ThreadPlanCallFunctionUsingABI.cpp
  ThreadPlanCallOnFunctionExit.cpp
  ThreadPlanCallUserExpression.cpp
  ThreadPlanPython.cpp
  ThreadPlanRunToAddress.cpp
  ThreadPlanShouldStopHere.cpp
  ThreadPlanStepInRange.cpp
  ThreadPlanStepInstruction.cpp
  ThreadPlanStepOut.cpp
  ThreadPlanStepOverBreakpoint.cpp
  ThreadPlanStepOverRange.cpp
  ThreadPlanStepRange.cpp
  ThreadPlanStepThrough.cpp
  ThreadPlanStepUntil.cpp
  ThreadPlanTracer.cpp
  ThreadSpec.cpp
  UnixSignals.cpp
  UnwindAssembly.cpp

  LINK_LIBS
    swiftAST
    swiftBasic
    swiftFrontend
    swiftRemoteAST
    ${REFLECTION_LIB}
    lldbBreakpoint
    lldbCore
    lldbExpression
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbUtility
    lldbPluginExpressionParserClang
    lldbPluginExpressionParserSwift
    lldbPluginObjCLanguage
    lldbPluginSwiftLanguage
    lldbPluginProcessUtility

  LINK_COMPONENTS
    Support
  )
