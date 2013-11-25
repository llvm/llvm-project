if (WIN32)
  set(prop_file_in "Microsoft.Cpp.Win32.llvm.props.in")
  set(prop_file_v100 "Microsoft.Cpp.Win32.LLVM-vs2010.props")
  set(prop_file_v110 "Microsoft.Cpp.Win32.LLVM-vs2012.props")
  set(prop_file_v110_xp "Microsoft.Cpp.Win32.LLVM-vs2012_xp.props")
  set(prop_file_v120 "toolset-vs2013.props")
  set(prop_file_v120_xp "toolset-vs2013_xp.props")

  # CPack will install a registry key in this format that we wish to reference.
  set(REG_KEY "${CPACK_PACKAGE_INSTALL_REGISTRY_KEY}")

  set(VS_VERSION "v100")
  set(MSC_VERSION "1600")
  configure_file(${prop_file_in} ${prop_file_v100})
  set(VS_VERSION "v110")
  set(MSC_VERSION "1700")
  configure_file(${prop_file_in} ${prop_file_v110})
  set(VS_VERSION "v110_xp")
  configure_file(${prop_file_in} ${prop_file_v110_xp})
  set(VS_VERSION "v120")
  set(MSC_VERSION "1800")
  configure_file(${prop_file_in} ${prop_file_v120})
  set(VS_VERSION "v120_xp")
  configure_file(${prop_file_in} ${prop_file_v120_xp})

  set(REG_KEY)
  set(VS_VERSION)
  set(MSC_VERSION)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${prop_file_v100}" DESTINATION tools/msbuild)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${prop_file_v110}" DESTINATION tools/msbuild)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${prop_file_v110_xp}" DESTINATION tools/msbuild)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${prop_file_v120}" DESTINATION tools/msbuild)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${prop_file_v120_xp}" DESTINATION tools/msbuild)

  install(DIRECTORY .
    DESTINATION tools/msbuild
    FILES_MATCHING
    PATTERN "*.targets"
    PATTERN "*.bat"
    PATTERN ".svn" EXCLUDE
    )
endif()
