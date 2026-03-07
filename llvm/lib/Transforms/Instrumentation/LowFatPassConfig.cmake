if(LOWFAT_SIZES_CFG)
  # LLVM pass needs the generated header too. Since runtimes build AFTER LLVM,
  # we must generate a local copy of the header for the pass to use.
  #
  # We use CMAKE_C_COMPILER (not add_executable) so that the generator is
  # always built for the host machine. add_executable would produce a
  # target-architecture binary on cross-compile setups (e.g. building for
  # AArch64 on an x86 host) which cannot be executed during the build.
  set(LOWFAT_CONFIG_GEN_SRC
      ${LLVM_MAIN_SRC_DIR}/../compiler-rt/lib/lowfat/tools/lf_config_gen.c)
  set(LOWFAT_CONFIG_GEN_BIN
      ${CMAKE_CURRENT_BINARY_DIR}/lf_config_gen_llvm${CMAKE_EXECUTABLE_SUFFIX})
  set(LOWFAT_GENERATED_HEADER ${CMAKE_CURRENT_BINARY_DIR}/lf_config_generated.h)

  add_custom_command(
    OUTPUT  ${LOWFAT_CONFIG_GEN_BIN}
    COMMAND ${CMAKE_C_COMPILER} -std=c11 -O2
            ${LOWFAT_CONFIG_GEN_SRC}
            -o ${LOWFAT_CONFIG_GEN_BIN}
    DEPENDS ${LOWFAT_CONFIG_GEN_SRC}
    COMMENT "Compiling lf_config_gen host tool (for LLVM pass)"
    VERBATIM
  )

  add_custom_command(
    OUTPUT  ${LOWFAT_GENERATED_HEADER}
    COMMAND ${LOWFAT_CONFIG_GEN_BIN} ${LOWFAT_SIZES_CFG} ${LOWFAT_GENERATED_HEADER}
    DEPENDS ${LOWFAT_CONFIG_GEN_BIN} ${LOWFAT_SIZES_CFG}
    COMMENT "Generating LowFat size config for LLVM pass"
    VERBATIM
  )

  add_custom_target(lowfat_pass_config DEPENDS ${LOWFAT_GENERATED_HEADER})
  add_dependencies(LLVMInstrumentation lowfat_pass_config)

  set_property(SOURCE LowFatSanitizer.cpp APPEND PROPERTY COMPILE_DEFINITIONS "LOWFAT_CUSTOM_CONFIG=1")
  set_property(SOURCE LowFatSanitizer.cpp APPEND PROPERTY OBJECT_DEPENDS ${LOWFAT_GENERATED_HEADER})

  # Tell the compiler where to find the generated header.
  target_include_directories(LLVMInstrumentation PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
endif()
