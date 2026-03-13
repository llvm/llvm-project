set_property(DIRECTORY PROPERTY LLDB_PLUGIN_KIND Trace)

option(LLDB_BUILD_INTEL_PT "Enable Building of Intel(R) Processor Trace Tool" OFF)
option(LLDB_BUILD_ARM_ETM "Enable Building of ARM Embedded Trace Macrocell Tool" OFF)

add_subdirectory(common)

if (LLDB_BUILD_INTEL_PT)
  add_subdirectory(intel-pt)
endif()

if (LLDB_BUILD_ARM_ETM)
  add_subdirectory(arm-etm)
endif()
