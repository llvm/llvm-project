set(SWIFT_SOURCES HostInfoMacOSXSwift.cpp)
set(LLVM_OPTIONAL_SOURCES ${SWIFT_SOURCES})

remove_module_flags()
include_directories(..)

add_lldb_library(lldbHostMacOSXObjCXX NO_PLUGIN_DEPENDENCIES
  Host.mm
  HostInfoMacOSX.mm
  HostThreadMacOSX.mm
  MemoryMonitorMacOSX.mm

  ## BEGIN SWIFT
  ${SWIFT_SOURCES}
  ## END SWIFT

  ADDITIONAL_HEADER_DIRS
    ${LLDB_INCLUDE_DIR}/lldb/Host/macosx/objcxx
  LINK_COMPONENTS
    Support
    TargetParser
  LINK_LIBS
    lldbCore
    lldbUtility
    ${EXTRA_LIBS}
  )

target_compile_options(lldbHostMacOSXObjCXX PRIVATE
  -fno-objc-exceptions
  -Wno-deprecated-declarations)
