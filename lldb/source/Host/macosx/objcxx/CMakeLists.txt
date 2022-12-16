set(SWIFT_SOURCES HostInfoMacOSXSwift.cpp)
set(LLVM_OPTIONAL_SOURCES ${SWIFT_SOURCES})
if (NOT LLDB_ENABLE_SWIFT_SUPPORT)
  unset(SWIFT_SOURCES)
endif()

remove_module_flags()
include_directories(..)

add_lldb_library(lldbHostMacOSXObjCXX
  Host.mm
  HostInfoMacOSX.mm
  HostThreadMacOSX.mm

  ${SWIFT_SOURCES}

  LINK_LIBS
    lldbUtility
    ${EXTRA_LIBS}

  LINK_COMPONENTS
    Support
  )

target_compile_options(lldbHostMacOSXObjCXX PRIVATE -fno-objc-exceptions)
