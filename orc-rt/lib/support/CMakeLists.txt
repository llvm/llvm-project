# Support: general-purpose utilities shared by the libraries that ship. Bedrock
# embeds this library, and SPIRE will too.
#
# Support has no per-system sources, and should stay that way: SPIRE is
# JIT-linked on every execution of every JIT'd program, so every object here is
# weight it may have to carry. System APIs with out-of-line implementations
# belong to Bedrock, which SPIRE reaches through the C API instead. Where a
# system API is small enough to inline, it becomes a header under
# include/orc-rt-internal/support/ (see support/sys/CacheControl.h,
# support/sys/Errno.h and support/Environment.h).

set(ORC_RT_SUPPORT_SOURCES
  Error.cpp
  Logging.cpp
  RTTI.cpp
  )

# The printf logging backend needs a runtime implementation; the none backend
# is header-only and the os_log backend is not yet implemented.
if(ORC_RT_LOG_BACKEND STREQUAL "printf")
  list(APPEND ORC_RT_SUPPORT_SOURCES Logging_printf.cpp)
elseif(ORC_RT_LOG_BACKEND STREQUAL "os_log")
  list(APPEND ORC_RT_SUPPORT_SOURCES Logging_oslog.cpp)
endif()

# Support is always an object library: it is embedded into the libraries that
# ship, never shipped on its own.
add_library(orc-rt-support-objects OBJECT ${ORC_RT_SUPPORT_SOURCES})

set_target_properties(orc-rt-support-objects PROPERTIES
  POSITION_INDEPENDENT_CODE ${ORC_RT_ENABLE_PIC}
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

target_link_libraries(orc-rt-support-objects PUBLIC orc-rt-headers)

# Apply RTTI and exceptions compile flags
target_compile_options(orc-rt-support-objects PRIVATE ${ORC_RT_COMPILE_FLAGS})
