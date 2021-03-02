if ( LLVM_INCLUDE_UTILS )
  add_subdirectory(llvm-jitlink-executor)
endif()

set(LLVM_LINK_COMPONENTS
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  BinaryFormat
  ExecutionEngine
  JITLink
  MC
  Object
  OrcJIT
  OrcShared
  OrcTargetProcess
  RuntimeDyld
  Support
  )

add_llvm_tool(llvm-jitlink
  llvm-jitlink.cpp
  llvm-jitlink-elf.cpp
  llvm-jitlink-macho.cpp
  )

# export_executable_symbols() is a no-op on Windows if neither
# LLVM_EXPORTED_SYMBOL_FILE nor LLVM_EXPORT_SYMBOLS_FOR_PLUGINS are set, but
# the jitlink tests need llvm_orc_registerJITLoaderGDBWrapper to be exported
# from the executable to work.
# FIXME: Find a better workaround. Maybe this should use LLVM_EXPORTED_SYMBOL_FILE
# and an .exports file now that the binary has a required export.
if (WIN32)
  target_link_options(llvm-jitlink PRIVATE
    "/export:llvm_orc_registerJITLoaderGDBWrapper"
    )
endif()

export_executable_symbols(llvm-jitlink)
