set(LIBOMPTARGET_BUILD_AMDGPU_PLUGIN TRUE CACHE BOOL
  "Whether to build AMDGPU plugin")
set(LIBOMPTARGET_BUILD_CUDA_PLUGIN TRUE CACHE BOOL
  "Whether to build CUDA plugin")

set(PLUGINS_TEST_COMMON omptarget OMPT omptarget.devicertl)
set(PLUGINS_TEST_SOURCES NextgenPluginsTest.cpp)

if (LIBOMPTARGET_BUILD_AMDGPU_PLUGIN)
  libomptarget_say("Building plugin unit tests for AMDGPU")
  add_libompt_unittest(PluginsTestAmd ${PLUGINS_TEST_SOURCES})
  add_dependencies(PluginsTestAmd ${PLUGINS_TEST_COMMON} omptarget.rtl.amdgpu)
  target_link_libraries(PluginsTestAmd PRIVATE ${PLUGINS_TEST_COMMON} omptarget.rtl.amdgpu)
  target_include_directories(PluginsTestAmd PRIVATE ${LIBOMPTARGET_INCLUDE_DIR})
endif()

if (LIBOMPTARGET_BUILD_CUDA_PLUGIN)
  libomptarget_say("Building plugin unit tests for Cuda")
  add_libompt_unittest(PluginsTestCu ${PLUGINS_TEST_SOURCES})
  add_dependencies(PluginsTestCu ${PLUGINS_TEST_COMMON} omptarget.rtl.cuda)
  target_link_libraries(PluginsTestCu PRIVATE ${PLUGINS_TEST_COMMON} omptarget.rtl.cuda)
  target_include_directories(PluginsTestCu PRIVATE ${LIBOMPTARGET_INCLUDE_DIR})
endif()
