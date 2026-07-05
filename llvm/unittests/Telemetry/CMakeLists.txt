set(LLVM_LINK_COMPONENTS
  Telemetry
  Core
  Support
  )

add_llvm_unittest(TelemetryTests
  TelemetryTest.cpp
  )
