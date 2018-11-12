include_directories(../Plugins/Process/Utility)

set(REFLECTION_LIB "swiftReflection")

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    # CMAKE_SYSTEM_PROCESSOR is equivalent to `uname -p` on Linux.
    if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
      set(ARCH "x86_64")
    elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
      set(ARCH "aarch64")
    elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "ppc64")
      set(ARCH "powerpc64")
    elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "ppc64le")
      set(ARCH "powerpc64le")
    elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "s390x")
      set(ARCH "s390x")
    # FIXME: Only matches v6l/v7l - by far the most common variants
    elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv6l")
      set(ARCH "armv6")
    elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7l")
      set(ARCH "armv7")
    elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "AMD64")
      set(ARCH "x86_64")
    elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "IA64")
      set(ARCH "itanium")
    elseif("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "(x86|i686)")
      set(ARCH "i686")
    else()
      message(FATAL_ERROR "Unrecognized architecture on host system: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    set(REFLECTION_LIB "swiftReflection-linux-${ARCH}")
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
  StackFrameRecognizer.cpp
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
if(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  target_compile_options(lldbTarget PRIVATE -Wno-dollar-in-identifier-extension)
endif()
