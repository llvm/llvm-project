if (LLVM_ENABLE_PLUGINS)
  add_llvm_pass_plugin(pypass-plugin INSTALL_WITH_TOOLCHAIN
    pypass.cpp
    DEPENDS
    intrinsics_gen
  )

  install(TARGETS pypass-plugin
    LIBRARY DESTINATION "${LLVM_PLUGINS_INSTALL_DIR}"
  )

  # Don't add a 'lib' prefix. The plugin is no linker input.
  set_target_properties(pypass-plugin PROPERTIES
    PREFIX ""
  )
endif()
