lldb_tablegen(TargetProperties.inc -gen-lldb-property-defs
  SOURCE TargetProperties.td
  TARGET LLDBTargetPropertiesGen)

lldb_tablegen(TargetPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE TargetProperties.td
  TARGET LLDBTargetPropertiesEnumGen)

lldb_tablegen(TargetProperties.json -dump-json
  SOURCE TargetProperties.td
  TARGET LLDBTargetPropertiesJsonGen)

# TODO: Add property `NO_PLUGIN_DEPENDENCIES` to lldbTarget
add_lldb_library(lldbTarget
  ABI.cpp
  AssertFrameRecognizer.cpp
  BorrowedStackFrame.cpp
  CoreFileMemoryRanges.cpp
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
  MemoryRegionInfoCache.cpp
  MemoryTagMap.cpp
  ModuleCache.cpp
  OperatingSystem.cpp
  PathMappingList.cpp
  Platform.cpp
  Process.cpp
  ProcessIOHandler.cpp
  ProcessTrace.cpp
  Queue.cpp
  QueueItem.cpp
  QueueList.cpp
  RegisterContextUnwind.cpp
  RemoteAwarePlatform.cpp
  ScriptedThreadPlan.cpp
  SectionLoadHistory.cpp
  SectionLoadList.cpp
  StackFrame.cpp
  StackFrameList.cpp
  StackFrameRecognizer.cpp
  StackID.cpp
  Statistics.cpp
  StopInfo.cpp
  StructuredDataPlugin.cpp
  SyntheticFrameProvider.cpp
  SystemRuntime.cpp
  Target.cpp
  TargetAPIMutex.cpp
  TargetGroup.cpp
  TargetGroupList.cpp
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
  ThreadPlanRunToAddress.cpp
  ThreadPlanRunToBreakpoint.cpp
  ThreadPlanShouldStopHere.cpp
  ThreadPlanSingleThreadTimeout.cpp
  ThreadPlanStack.cpp
  ThreadPlanStepInRange.cpp
  ThreadPlanStepInstruction.cpp
  ThreadPlanStepOut.cpp
  ThreadPlanStepOverBreakpoint.cpp
  ThreadPlanStepOverRange.cpp
  ThreadPlanStepRange.cpp
  ThreadPlanStepThrough.cpp
  ThreadPlanStepUntil.cpp
  ThreadPlanTracer.cpp
  ThreadRegisterContext.cpp
  ThreadSpec.cpp
  Trace.cpp
  TraceCursor.cpp
  TraceDumper.cpp
  TraceExporter.cpp
  UnixSignals.cpp
  UnwindAssembly.cpp
  UnwindLLDB.cpp

  ADDITIONAL_HEADER_DIRS
    ${LLDB_INCLUDE_DIR}/lldb/Target
  LINK_COMPONENTS
    Support
    MC
  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbExpression
    lldbHost
    lldbInterpreter
    lldbPluginProcessUtility
    lldbRegister
    lldbSymbol
    lldbUtility
    lldbValueObject
    lldbVersion
  )

add_dependencies(lldbTarget
  LLDBTargetPropertiesGen
  LLDBTargetPropertiesEnumGen)
