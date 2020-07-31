# Flags to control each individual feature
option(LLDB_BUILD_INTEL_MPX "Enable Building of Intel(R) Memory Protection Extensions" ON)
option(LLDB_BUILD_INTEL_PT "Enable Building of Intel(R) Processor Trace Tool" OFF)

# Return if all features are OFF
if (NOT LLDB_BUILD_INTEL_MPX AND NOT LLDB_BUILD_INTEL_PT)
  return()
endif()

LIST (APPEND FEATURE_LIBS "")

# Add feature specific subdirectories based on flags
if (LLDB_BUILD_INTEL_MPX AND CMAKE_SYSTEM_NAME MATCHES "Linux")
  add_subdirectory(intel-mpx)
  LIST (APPEND FEATURE_LIBS ${FEATURE_LIBS} lldbIntelMPX)
  SET (CLI_WRAPPER_PREPROCESSORS "${CLI_WRAPPER_PREPROCESSORS} -DBUILD_INTEL_MPX")
endif()

if (LLDB_BUILD_INTEL_PT)
  add_subdirectory(intel-pt)
  LIST (APPEND FEATURE_LIBS ${FEATURE_LIBS} lldbIntelPT)
  SET (CLI_WRAPPER_PREPROCESSORS "${CLI_WRAPPER_PREPROCESSORS} -DBUILD_INTEL_PT")
endif()

# Add python wrapper if python not disabled
if (LLDB_ENABLE_PYTHON AND LLDB_BUILD_INTEL_PT)
  set(LLDB_INTEL_FEATURES_PYTHON_WRAP
     ${LLDB_BINARY_DIR}/tools/intel-features/scripts/IntelFeaturesPythonWrap.cpp)
  set_source_files_properties(${LLDB_INTEL_FEATURES_PYTHON_WRAP}
                              PROPERTIES GENERATED 1)

  if (CLANG_CL)
    set_source_files_properties(${LLDB_INTEL_FEATURES_PYTHON_WRAP}
                                 PROPERTIES COMPILE_FLAGS -Wno-unused-function)
  endif()

  if (LLVM_COMPILER_IS_GCC_COMPATIBLE AND
      NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
    set_property(SOURCE ${LLDB_INTEL_FEATURES_PYTHON_WRAP}
                 APPEND_STRING PROPERTY COMPILE_FLAGS
                 " -Wno-sequence-point -Wno-cast-qual")
  endif ()
  add_subdirectory(scripts)
endif()

if (NOT CLI_WRAPPER_PREPROCESSORS)
  return()
endif()

set_source_files_properties(cli-wrapper.cpp PROPERTIES
                            COMPILE_FLAGS ${CLI_WRAPPER_PREPROCESSORS})

add_lldb_library(lldbIntelFeatures SHARED
  cli-wrapper.cpp
  ${LLDB_INTEL_FEATURES_PYTHON_WRAP}

  LINK_LIBS
    ${FEATURE_LIBS}
    ${PYTHON_LIBRARIES}
  )

# Add link dependencies for python wrapper
if (LLDB_ENABLE_PYTHON AND LLDB_BUILD_INTEL_PT)
  add_dependencies(lldbIntelFeatures intel-features-swig_wrapper)
endif()

install(TARGETS lldbIntelFeatures
  LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX})
