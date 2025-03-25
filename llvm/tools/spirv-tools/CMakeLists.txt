option(LLVM_INCLUDE_SPIRV_TOOLS_TESTS "Include tests that use SPIRV-Tools" Off)
mark_as_advanced(LLVM_INCLUDE_SPIRV_TOOLS_TESTS)

if (NOT LLVM_INCLUDE_SPIRV_TOOLS_TESTS)
  return()
endif ()

if (NOT "SPIRV" IN_LIST LLVM_TARGETS_TO_BUILD)
  message(FATAL_ERROR "Building SPIRV-Tools tests is unsupported without the SPIR-V target")
endif ()

# SPIRV_DIS, SPIRV_VAL, SPIRV_AS and SPIRV_LINK variables can be used to provide paths to existing
# spirv-dis, spirv-val, spirv-as, and spirv-link binaries, respectively. Otherwise, build them from
# SPIRV-Tools source.
if (NOT SPIRV_DIS OR NOT SPIRV_VAL OR NOT SPIRV_AS OR NOT SPIRV_LINK)
  include(ExternalProject)

  set(BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/SPIRVTools-bin)

  ExternalProject_Add(SPIRVTools
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Tools.git
    GIT_TAG main
    BINARY_DIR ${BINARY_DIR}
    BUILD_COMMAND ${CMAKE_COMMAND} --build ${BINARY_DIR} --target spirv-dis spirv-val spirv-as spirv-link
    BUILD_BYPRODUCTS ${BINARY_DIR}/tools/spirv-dis ${BINARY_DIR}/tools/spirv-val ${BINARY_DIR}/tools/spirv-as ${BINARY_DIR}/tools/spirv-link
    DOWNLOAD_COMMAND git clone https://github.com/KhronosGroup/SPIRV-Tools.git SPIRVTools &&
      cd SPIRVTools &&
      ${Python3_EXECUTABLE} utils/git-sync-deps
    UPDATE_COMMAND git pull origin main &&
      ${Python3_EXECUTABLE} utils/git-sync-deps
    # Don't auto-update on every build.
    UPDATE_DISCONNECTED 1
    # Allow manual updating with an explicit SPIRVTools-update target.
    STEP_TARGETS update
    # Install handled below.
    INSTALL_COMMAND ""
    )
endif ()

if (CMAKE_HOST_UNIX)
  set(LLVM_LINK_OR_COPY create_symlink)
else ()
  set(LLVM_LINK_OR_COPY copy)
endif ()

# Link the provided or just built binaries.
if (SPIRV_DIS)
  add_custom_target(spirv-dis
    COMMAND ${CMAKE_COMMAND} -E ${LLVM_LINK_OR_COPY} "${SPIRV_DIS}" "${LLVM_RUNTIME_OUTPUT_INTDIR}/spirv-dis")
else ()
  add_custom_target(spirv-dis
    COMMAND ${CMAKE_COMMAND} -E ${LLVM_LINK_OR_COPY} "${BINARY_DIR}/tools/spirv-dis${CMAKE_EXECUTABLE_SUFFIX}" "${LLVM_RUNTIME_OUTPUT_INTDIR}/spirv-dis${CMAKE_EXECUTABLE_SUFFIX}"
    DEPENDS SPIRVTools
    )
endif ()

if (SPIRV_VAL)
  add_custom_target(spirv-val
    COMMAND ${CMAKE_COMMAND} -E ${LLVM_LINK_OR_COPY} "${SPIRV_VAL}" "${LLVM_RUNTIME_OUTPUT_INTDIR}/spirv-val")
else ()
  add_custom_target(spirv-val
    COMMAND ${CMAKE_COMMAND} -E ${LLVM_LINK_OR_COPY} "${BINARY_DIR}/tools/spirv-val${CMAKE_EXECUTABLE_SUFFIX}" "${LLVM_RUNTIME_OUTPUT_INTDIR}/spirv-val${CMAKE_EXECUTABLE_SUFFIX}"
    DEPENDS SPIRVTools
    )
endif ()

if (SPIRV_AS)
  add_custom_target(spirv-as
    COMMAND ${CMAKE_COMMAND} -E ${LLVM_LINK_OR_COPY} "${SPIRV_AS}" "${LLVM_RUNTIME_OUTPUT_INTDIR}/spirv-as")
else ()
  add_custom_target(spirv-as
    COMMAND ${CMAKE_COMMAND} -E ${LLVM_LINK_OR_COPY} "${BINARY_DIR}/tools/spirv-as${CMAKE_EXECUTABLE_SUFFIX}" "${LLVM_RUNTIME_OUTPUT_INTDIR}/spirv-as${CMAKE_EXECUTABLE_SUFFIX}"
    DEPENDS SPIRVTools
    )
endif ()

if (SPIRV_LINK)
  add_custom_target(spirv-link
    COMMAND ${CMAKE_COMMAND} -E ${LLVM_LINK_OR_COPY} "${SPIRV_LINK}" "${LLVM_RUNTIME_OUTPUT_INTDIR}/spirv-link")
else ()
  add_custom_target(spirv-link
    COMMAND ${CMAKE_COMMAND} -E ${LLVM_LINK_OR_COPY} "${BINARY_DIR}/tools/spirv-link${CMAKE_EXECUTABLE_SUFFIX}" "${LLVM_RUNTIME_OUTPUT_INTDIR}/spirv-link${CMAKE_EXECUTABLE_SUFFIX}"
    DEPENDS SPIRVTools
    )
endif ()
