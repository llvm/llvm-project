# Set the correct rpath to locate libswiftCore.
if( CMAKE_SYSTEM_NAME MATCHES "Linux" )
  if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "ppc64le")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} \
        -Wl,-rpath,${CMAKE_BINARY_DIR}/../swift-linux-powerpc64le/lib${LLVM_LIBDIR_SUFFIX}/swift/linux/powerpc64le")
  else()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} \
        -Wl,-rpath,${CMAKE_BINARY_DIR}/../swift-linux-x86_64/lib${LLVM_LIBDIR_SUFFIX}/swift/linux")
  endif()
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib/swift/linux:${CMAKE_INSTALL_RPATH}")
endif()

# Only set LLVM_CODESIGNING_IDENTITY for building on Apple hosts for Apple
# targets
if (CMAKE_HOST_APPLE AND APPLE)
  # Override locally, so the repl is ad-hoc signed.
  set(LLVM_CODESIGNING_IDENTITY "-")
endif()

# Requires system-provided Swift libs.
set(CMAKE_OSX_DEPLOYMENT_TARGET 10.14.4)

add_lldb_tool(repl_swift ADD_TO_FRAMEWORK
  main.c
  )
target_link_libraries(repl_swift PRIVATE ${CMAKE_DL_LIBS})
if(CMAKE_SYSTEM_NAME STREQUAL Windows)
  set_target_properties(repl_swift PROPERTIES
    WIN32_EXECUTABLE TRUE)
endif()

# The dummy repl executable is a C program, but we always look for a mangled
# swift symbol (corresponding to main). If we build the repl with debug info,
# the debugger looks at the frame language (looking up the compile unit) and gets
# confused.
if(MSVC)
  # disable inline function expansion so that we have a function that we can
  # break upon
  target_compile_options(repl_swift PRIVATE /Ob0)
else()
  set_target_properties(repl_swift PROPERTIES
      COMPILE_FLAGS "-g0")
endif()

if(APPLE)
  # Set the RPATHs to locate libswiftCore. Prefer the just-built one.
  set(swift_buildtree ${SWIFT_BINARY_DIR}/lib/swift/macosx)
  set(system_libs /usr/lib/swift)

  if(LLDB_BUILD_FRAMEWORK)
    get_target_property(framework_build_dir liblldb LIBRARY_OUTPUT_DIRECTORY)
    set(lldb_buildtree ${framework_build_dir}/LLDB.framework/Versions/${LLDB_FRAMEWORK_VERSION}/Resources/Swift/macosx)
    set(lldb_installtree Swift/macosx) # repl_swift is installed to framework Resources
  endif()

  lldb_setup_rpaths(repl_swift
    BUILD_RPATH
      ${swift_buildtree}
      ${lldb_buildtree}
      ${system_libs}
    INSTALL_RPATH
      ${lldb_installtree}
      ${system_libs}
  )
endif()
