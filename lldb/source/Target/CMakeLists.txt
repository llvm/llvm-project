lldb_tablegen(TargetProperties.inc -gen-lldb-property-defs
  SOURCE TargetProperties.td
  TARGET LLDBTargetPropertiesGen)

lldb_tablegen(TargetPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE TargetProperties.td
  TARGET LLDBTargetPropertiesEnumGen)

add_lldb_library(lldbTarget
  ABI.cpp
  AssertFrameRecognizer.cpp
  ExecutionContext.cpp
  JITLoader.cpp
  JITLoaderList.cpp
  InstrumentationRuntime.cpp
  InstrumentationRuntimeStopInfo.cpp
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
  SwiftLanguageRuntime.cpp
  SwiftLanguageRuntimeDynamicTypeResolution.cpp
  SwiftLanguageRuntimeNames.cpp
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

  LINK_LIBS
    swiftAST
    swiftBasic
    swiftReflection
    swiftRemoteAST
    lldbBreakpoint
    lldbCore
    lldbExpression
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbUtility
    lldbPluginExpressionParserClang
    lldbPluginExpressionParserSwift
    lldbPluginSwiftLanguage
    lldbPluginProcessUtility

  LINK_COMPONENTS
    Support
  )
if(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  target_compile_options(lldbTarget PRIVATE -Wno-dollar-in-identifier-extension)
elseif(MSVC)
  target_compile_options(lldbTarget PRIVATE /permissive-)
endif()

add_dependencies(lldbTarget
  LLDBTargetPropertiesGen
  LLDBTargetPropertiesEnumGen)

# Necessary to ensure that the SWIFT_INLINE_NAMESPACE macro is defined for
# swiftDemangling inclusions.
target_compile_definitions(lldbTarget PRIVATE
  SWIFT_INLINE_NAMESPACE=compiler)
