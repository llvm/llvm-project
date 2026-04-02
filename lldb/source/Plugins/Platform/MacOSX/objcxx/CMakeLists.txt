remove_module_flags()
include_directories(.)

# ObjC++ (.mm) files cannot be unity-built because CMake generates .cxx
# unity wrappers, causing them to be compiled as C++ instead of ObjC++.
set(CMAKE_UNITY_BUILD OFF)

add_lldb_library(lldbPluginPlatformMacOSXObjCXX
  PlatformiOSSimulatorCoreSimulatorSupport.mm

  LINK_COMPONENTS
    Object
    Support
  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbUtility
    ${EXTRA_LIBS}
  )


target_compile_options(lldbPluginPlatformMacOSXObjCXX PRIVATE -fno-objc-exceptions)
