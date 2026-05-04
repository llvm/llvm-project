remove_module_flags()
include_directories(..)

find_library(SECURITY_FRAMEWORK Security)

add_lldb_library(lldbHostMacOSXObjCXX NO_PLUGIN_DEPENDENCIES
  Host.mm
  HostInfoMacOSX.mm
  HostThreadMacOSX.mm
  MemoryMonitorMacOSX.mm

  ADDITIONAL_HEADER_DIRS
    ${LLDB_INCLUDE_DIR}/lldb/Host/macosx/objcxx
  LINK_COMPONENTS
    Support
    TargetParser
  LINK_LIBS
    lldbUtility
    ${EXTRA_LIBS}
    ${SECURITY_FRAMEWORK}
  )

target_compile_options(lldbHostMacOSXObjCXX PRIVATE
  -fno-objc-exceptions
  -Wno-deprecated-declarations)
