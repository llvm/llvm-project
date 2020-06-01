lldb_tablegen(TargetProperties.inc -gen-lldb-property-defs
  SOURCE TargetProperties.td
  TARGET LLDBTargetPropertiesGen)

lldb_tablegen(TargetPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE TargetProperties.td
  TARGET LLDBTargetPropertiesEnumGen)

set(SWIFT_SOURCES
  SwiftLanguageRuntime.cpp
  SwiftLanguageRuntimeDynamicTypeResolution.cpp
  SwiftLanguageRuntimeNames.cpp
)
set(SWIFT_LIBS
    swiftAST
    swiftBasic
    swiftReflection
    swiftRemoteAST
    lldbPluginExpressionParserSwift
    lldbPluginSwiftLanguage
)
set(LLVM_OPTIONAL_SOURCES ${SWIFT_SOURCES})
if (NOT LLDB_ENABLE_SWIFT_SUPPORT)
  unset(SWIFT_SOURCES)
  unset(SWIFT_LIBS)
endif()

add_lldb_library(lldbTarget
  ABI.cpp
  AssertFrameRecognizer.cpp
  ExecutionContext.cpp
  InstrumentationRuntime.cpp
  InstrumentationRuntimeStopInfo.cpp
  JITLoader.cpp
  JITLoaderList.cpp
  Language.cpp
  LanguageRuntime.cpp
  Memory.cpp
  MemoryHistory.cpp
  MemoryRegionInfo.cpp
  ModuleCache.cpp
  OperatingSystem.cpp
  PathMappingList.cpp
  Platform.cpp
  Process.cpp
  Queue.cpp
  QueueItem.cpp
  QueueList.cpp
  RegisterContext.cpp
  RegisterContextUnwind.cpp
  RegisterNumber.cpp
  RemoteAwarePlatform.cpp
  SectionLoadHistory.cpp
  SectionLoadList.cpp
  StackFrame.cpp
  StackFrameList.cpp
  StackFrameRecognizer.cpp
  StackID.cpp
  StopInfo.cpp
  StructuredDataPlugin.cpp
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
  ThreadPlanStack.cpp
  ThreadSpec.cpp
  UnixSignals.cpp
  UnwindAssembly.cpp
  UnwindLLDB.cpp

  ${SWIFT_SOURCES}

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbExpression
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbUtility
    lldbPluginExpressionParserClang
    lldbPluginProcessUtility

    ${SWIFT_LIBS}

  LINK_COMPONENTS
    Support
    MC
  )
if(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  target_compile_options(lldbTarget PRIVATE -Wno-dollar-in-identifier-extension)
elseif(MSVC)
  target_compile_options(lldbTarget PRIVATE /permissive-)
endif()

add_dependencies(lldbTarget
  LLDBTargetPropertiesGen
  LLDBTargetPropertiesEnumGen)
