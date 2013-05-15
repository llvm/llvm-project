set(LLVM_NO_RTTI 1)

add_lldb_library(lldbPluginAppleObjCRuntime
  AppleObjCRuntime.cpp
  AppleObjCRuntimeV1.cpp
  AppleObjCRuntimeV2.cpp
  AppleObjCTrampolineHandler.cpp
  AppleObjCTypeVendor.cpp
  AppleThreadPlanStepThroughObjCTrampoline.cpp
  )
