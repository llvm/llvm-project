# AddLLVMSwiftDriver.cmake
#
# Defines llvm_add_swift_driver_external_project(root) which creates
# ExternalProject targets that build swift-driver and its dependencies
# (llbuild, swift-tools-support-core, swift-argument-parser) using the
# HOST Swift compiler.  The finished swift-driver binary is installed
# alongside swiftc in ${CMAKE_BINARY_DIR}/bin/.
function(llvm_add_swift_driver_external_project root)
  include(ExternalProject)

  # -------------------------------------------------------------------------
  # Locate the host Swift compiler.
  #
  # On Apple platforms the default Xcode toolchain swiftc may be a different
  # version than the active SDK requires.  Xcode ships a platform-specific
  # toolchain (e.g. OSX27.0.xctoolchain) that matches the SDK; prefer that
  # over whatever `swiftc` resolves to on PATH or in the cmake cache.
  #
  # We use xcrun to get the real SDK path (CMAKE_OSX_SYSROOT may be a short
  # name like "macosx" rather than the full path) and derive the toolchain
  # from the SDK version encoded in that path.
  # -------------------------------------------------------------------------
  set(_host_swiftc "")
  if(APPLE)
    execute_process(
      COMMAND xcrun --show-sdk-path
      OUTPUT_VARIABLE _sdk_path
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    execute_process(
      COMMAND xcode-select -p
      OUTPUT_VARIABLE _xdev
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(_sdk_path MATCHES "MacOSX([0-9]+\\.[0-9]+)\\.sdk")
      set(_candidate
          "${_xdev}/Toolchains/OSX${CMAKE_MATCH_1}.xctoolchain/usr/bin/swiftc")
      if(EXISTS "${_candidate}")
        set(_host_swiftc "${_candidate}")
      endif()
    endif()
  endif()
  if(NOT _host_swiftc)
    find_program(_host_swiftc_found swiftc)
    set(_host_swiftc "${_host_swiftc_found}")
  endif()

  if(NOT _host_swiftc OR _host_swiftc STREQUAL "_host_swiftc_found-NOTFOUND")
    message(STATUS "swift-driver: host swiftc not found – skipping pre-built swift-driver")
    return()
  endif()
  message(STATUS "swift-driver: building with host swiftc: ${_host_swiftc}")

  # -------------------------------------------------------------------------
  # Source and build directories.
  # -------------------------------------------------------------------------
  set(_driver_src   "${root}/swift-driver")
  set(_llbuild_src  "${root}/llbuild")
  set(_tsc_src      "${root}/swift-tools-support-core")
  set(_argp_src     "${root}/swift-argument-parser")
  set(_host_build   "${CMAKE_BINARY_DIR}/swift-driver-host")

  foreach(_src IN ITEMS _driver_src _llbuild_src _tsc_src _argp_src)
    if(NOT EXISTS "${${_src}}")
      message(STATUS "swift-driver: ${${_src}} not found – skipping pre-built swift-driver")
      return()
    endif()
  endforeach()

  # cmake config dirs inside each ExternalProject's build tree (mirrors the
  # pattern all four projects use: configure_file(…Config.cmake.in
  #   ${CMAKE_CURRENT_BINARY_DIR}/…Config.cmake) under cmake/modules/).
  set(_llbuild_cmake  "${_host_build}/llbuild/cmake/modules")
  set(_tsc_cmake      "${_host_build}/tsc/cmake/modules")
  set(_argp_cmake     "${_host_build}/argparser/cmake/modules")

  set(_driver_bin     "${_host_build}/swift-driver/bin/swift-driver")
  set(_driver_dst     "${CMAKE_BINARY_DIR}/bin/swift-driver-new")

  # Common cmake flags forwarded to every sub-build.
  # On Apple we pin the sysroot and deployment target explicitly so that the
  # platform-specific swiftc (which may default to a newer OS version than the
  # installed SDK covers) can find the Swift standard library.
  set(_common
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_Swift_COMPILER=${_host_swiftc}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  )
  if(APPLE AND _sdk_path)
    list(APPEND _common -DCMAKE_OSX_SYSROOT=${_sdk_path})
  endif()
  if(APPLE)
    set(_deploy_target "${CMAKE_OSX_DEPLOYMENT_TARGET}")
    if(NOT _deploy_target)
      set(_deploy_target "13.0")
    endif()
    list(APPEND _common -DCMAKE_OSX_DEPLOYMENT_TARGET=${_deploy_target})
  endif()

  # -------------------------------------------------------------------------
  # llbuild
  # -------------------------------------------------------------------------
  ExternalProject_Add(swift-driver-dep-llbuild
    SOURCE_DIR  "${_llbuild_src}"
    BINARY_DIR  "${_host_build}/llbuild"
    CMAKE_ARGS
      ${_common}
      -DLLBUILD_SUPPORT_BINDINGS=Swift   # swift-driver needs llbuildSwift
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${_llbuild_cmake}/LLBuildConfig.cmake"
  )

  # -------------------------------------------------------------------------
  # swift-argument-parser
  # -------------------------------------------------------------------------
  ExternalProject_Add(swift-driver-dep-argparser
    SOURCE_DIR  "${_argp_src}"
    BINARY_DIR  "${_host_build}/argparser"
    CMAKE_ARGS
      ${_common}
      -DBUILD_EXAMPLES=OFF
      -DBUILD_TESTING=OFF
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${_argp_cmake}/ArgumentParserConfig.cmake"
  )

  # -------------------------------------------------------------------------
  # swift-tools-support-core (depends on llbuild)
  # -------------------------------------------------------------------------
  ExternalProject_Add(swift-driver-dep-tsc
    SOURCE_DIR  "${_tsc_src}"
    BINARY_DIR  "${_host_build}/tsc"
    CMAKE_ARGS
      ${_common}
      -DLLBuild_DIR=${_llbuild_cmake}
    DEPENDS swift-driver-dep-llbuild
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${_tsc_cmake}/TSCConfig.cmake"
  )

  # -------------------------------------------------------------------------
  # swift-driver
  # -------------------------------------------------------------------------
  ExternalProject_Add(swift-driver-prebuilt
    SOURCE_DIR  "${_driver_src}"
    BINARY_DIR  "${_host_build}/swift-driver"
    CMAKE_ARGS
      ${_common}
      -DLLBuild_DIR=${_llbuild_cmake}
      -DTSC_DIR=${_tsc_cmake}
      -DArgumentParser_DIR=${_argp_cmake}
    DEPENDS
      swift-driver-dep-llbuild
      swift-driver-dep-tsc
      swift-driver-dep-argparser
    # Copy the finished binary next to swiftc so the runtime exec finds it.
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
        "${_driver_bin}"
        "${_driver_dst}"
    BUILD_BYPRODUCTS "${_driver_bin}" "${_driver_dst}"
  )
endfunction()


get_filename_component(_swift_driver_root "${LLVM_EXTERNAL_SWIFT_SOURCE_DIR}" DIRECTORY)
llvm_add_swift_driver_external_project("${_swift_driver_root}")
add_llvm_external_project(swift)

if(TARGET swift-driver-prebuilt AND TARGET swift-frontend)
  add_dependencies(swift-frontend swift-driver-prebuilt)
endif()
