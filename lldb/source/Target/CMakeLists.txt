include_directories(../Plugins/Process/Utility)

set(REFLECTION_LIB "swiftReflection")

# Duplicated from SwiftSupport.cmake in
# swift-corelibs-{libdispatch,Foundation,XCTest}
function(get_swift_host_arch result_var_name)
  if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
    set("${result_var_name}" "x86_64" PARENT_SCOPE)
  elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
    set("${result_var_name}" "aarch64" PARENT_SCOPE)
  elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "ppc64")
    set("${result_var_name}" "powerpc64" PARENT_SCOPE)
  elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "ppc64le")
    set("${result_var_name}" "powerpc64le" PARENT_SCOPE)
  elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "s390x")
    set("${result_var_name}" "s390x" PARENT_SCOPE)
  elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv6l")
    set("${result_var_name}" "armv6" PARENT_SCOPE)
  elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7l")
    set("${result_var_name}" "armv7" PARENT_SCOPE)
  elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "AMD64")
    set("${result_var_name}" "x86_64" PARENT_SCOPE)
  elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "IA64")
    set("${result_var_name}" "itanium" PARENT_SCOPE)
  elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86")
    set("${result_var_name}" "i686" PARENT_SCOPE)
  else()
    message(FATAL_ERROR "Unrecognized architecture on host system: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
endfunction()

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
else()
  string(TOLOWER ${CMAKE_SYSTEM_NAME} swift_os)
  get_swift_host_arch(swift_arch)
  set(REFLECTION_LIB "swiftReflection-${swift_os}-${swift_arch}")
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
