# The lane lowering is pure graph code, so it is compiled straight into the test
# rather than linked from the plugin archive: no device, no HSA, no libomptarget.
# OffloadError.cpp comes along only for the error class the scheduler reports
# through.
set(LLVM_LINK_COMPONENTS Support)

add_unittest(OffloadUnitTests "TaskGraph.unittests"
  TaskGraphScheduleTest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../plugins-nextgen/amdgpu/src/TaskGraphSchedule.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../plugins-nextgen/common/src/OffloadError.cpp
)

add_dependencies(TaskGraph.unittests PluginErrcodes)

target_include_directories(TaskGraph.unittests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../plugins-nextgen/amdgpu/src
  $<TARGET_PROPERTY:PluginCommon,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_definitions(TaskGraph.unittests PRIVATE
  TARGET_NAME=TaskGraphSchedule
  DEBUG_PREFIX="TaskGraphSchedule"
)
