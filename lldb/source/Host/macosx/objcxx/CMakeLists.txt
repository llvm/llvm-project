
remove_module_flags()
include_directories(..)

# ObjC++ (.mm) files cannot be unity-built because CMake generates .cxx
# unity wrappers, causing them to be compiled as C++ instead of ObjC++.
set(CMAKE_UNITY_BUILD OFF)

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
  )

target_compile_options(lldbHostMacOSXObjCXX PRIVATE
  -fno-objc-exceptions
  -Wno-deprecated-declarations)
