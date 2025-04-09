add_subdirectory(ClangCommon)

# Preserve language plug-ins sorting.
add_subdirectory(CPlusPlus)
add_subdirectory(ObjC)
add_subdirectory(ObjCPlusPlus)
if(LLDB_ENABLE_SWIFT_SUPPORT)
  add_subdirectory(Swift)
endif()
