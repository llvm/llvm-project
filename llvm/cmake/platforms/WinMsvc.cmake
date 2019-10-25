# Cross toolchain configuration for using clang-cl on non-Windows hosts to
# target MSVC.
#
# Usage:
# cmake -G Ninja
#    -DCMAKE_TOOLCHAIN_FILE=/path/to/this/file
#    -DHOST_ARCH=[aarch64|arm64|armv7|arm|i686|x86|x86_64|x64]
#    -DLLVM_NATIVE_TOOLCHAIN=/path/to/llvm/installation
#    -DMSVC_BASE=/path/to/MSVC/system/libraries/and/includes
#    -DWINSDK_BASE=/path/to/windows-sdk
#    -DWINSDK_VER=windows sdk version folder name
#
# HOST_ARCH:
#    The architecture to build for.
#
# LLVM_NATIVE_TOOLCHAIN:
#   *Absolute path* to a folder containing the toolchain which will be used to
#   build.  At a minimum, this folder should have a bin directory with a
#   copy of clang-cl, clang, clang++, and lld-link, as well as a lib directory
#   containing clang's system resource directory.
#
# MSVC_BASE:
#   *Absolute path* to the folder containing MSVC headers and system libraries.
#   The layout of the folder matches that which is intalled by MSVC 2017 on
#   Windows, and should look like this:
#
# ${MSVC_BASE}
#   include
#     vector
#     stdint.h
#     etc...
#   lib
#     x64
#       libcmt.lib
#       msvcrt.lib
#       etc...
#     x86
#       libcmt.lib
#       msvcrt.lib
#       etc...
#
# For versions of MSVC < 2017, or where you have a hermetic toolchain in a
# custom format, you must use symlinks or restructure it to look like the above.
#
# WINSDK_BASE:
#   Together with WINSDK_VER, determines the location of Windows SDK headers
#   and libraries.
#
# WINSDK_VER:
#   Together with WINSDK_BASE, determines the locations of Windows SDK headers
#   and libraries.
#
# WINSDK_BASE and WINSDK_VER work together to define a folder layout that matches
# that of the Windows SDK installation on a standard Windows machine.  It should
# match the layout described below.
#
# Note that if you install Windows SDK to a windows machine and simply copy the
# files, it will already be in the correct layout.
#
# ${WINSDK_BASE}
#   Include
#     ${WINSDK_VER}
#       shared
#       ucrt
#       um
#         windows.h
#         etc...
#   Lib
#     ${WINSDK_VER}
#       ucrt
#         x64
#         x86
#           ucrt.lib
#           etc...
#       um
#         x64
#         x86
#           kernel32.lib
#           etc
#
# IMPORTANT: In order for this to work, you will need a valid copy of the Windows
# SDK and C++ STL headers and libraries on your host.  Additionally, since the
# Windows libraries and headers are not case-correct, this toolchain file sets
# up a VFS overlay for the SDK headers and case-correcting symlinks for the
# libraries when running on a case-sensitive filesystem.


# When configuring CMake with a toolchain file against a top-level CMakeLists.txt,
# it will actually run CMake many times, once for each small test program used to
# determine what features a compiler supports.  Unfortunately, none of these
# invocations share a CMakeCache.txt with the top-level invocation, meaning they
# won't see the value of any arguments the user passed via -D.  Since these are
# necessary to properly configure MSVC in both the top-level configuration as well as
# all feature-test invocations, we set environment variables with the values so that
# these environments get inherited by child invocations. We can switch to
# CMAKE_TRY_COMPILE_PLATFORM_VARIABLES once our minimum supported CMake version
# is 3.6 or greater.
function(init_user_prop prop)
  if(${prop})
    set(ENV{_${prop}} "${${prop}}")
  else()
    set(${prop} "$ENV{_${prop}}" PARENT_SCOPE)
  endif()
endfunction()

function(generate_winsdk_vfs_overlay winsdk_include_dir output_path)
  set(include_dirs)
  file(GLOB_RECURSE entries LIST_DIRECTORIES true "${winsdk_include_dir}/*")
  foreach(entry ${entries})
    if(IS_DIRECTORY "${entry}")
      list(APPEND include_dirs "${entry}")
    endif()
  endforeach()

  file(WRITE "${output_path}"  "version: 0\n")
  file(APPEND "${output_path}" "case-sensitive: false\n")
  file(APPEND "${output_path}" "roots:\n")

  foreach(dir ${include_dirs})
    file(GLOB headers RELATIVE "${dir}" "${dir}/*.h")
    if(NOT headers)
      continue()
    endif()

    file(APPEND "${output_path}" "  - name: \"${dir}\"\n")
    file(APPEND "${output_path}" "    type: directory\n")
    file(APPEND "${output_path}" "    contents:\n")

    foreach(header ${headers})
      file(APPEND "${output_path}" "      - name: \"${header}\"\n")
      file(APPEND "${output_path}" "        type: file\n")
      file(APPEND "${output_path}" "        external-contents: \"${dir}/${header}\"\n")
    endforeach()
  endforeach()
endfunction()

function(generate_winsdk_lib_symlinks winsdk_um_lib_dir output_dir)
  execute_process(COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_dir}")
  file(GLOB libraries RELATIVE "${winsdk_um_lib_dir}" "${winsdk_um_lib_dir}/*")
  foreach(library ${libraries})
    string(TOLOWER "${library}" all_lowercase_symlink_name)
    if(NOT library STREQUAL all_lowercase_symlink_name)
      execute_process(COMMAND "${CMAKE_COMMAND}"
                              -E create_symlink
                              "${winsdk_um_lib_dir}/${library}"
                              "${output_dir}/${all_lowercase_symlink_name}")
    endif()

    get_filename_component(name_we "${library}" NAME_WE)
    get_filename_component(ext "${library}" EXT)
    string(TOLOWER "${ext}" lowercase_ext)
    set(lowercase_ext_symlink_name "${name_we}${lowercase_ext}")
    if(NOT library STREQUAL lowercase_ext_symlink_name AND
       NOT all_lowercase_symlink_name STREQUAL lowercase_ext_symlink_name)
      execute_process(COMMAND "${CMAKE_COMMAND}"
                              -E create_symlink
                              "${winsdk_um_lib_dir}/${library}"
                              "${output_dir}/${lowercase_ext_symlink_name}")
    endif()
  endforeach()
endfunction()

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_VERSION 10.0)
set(CMAKE_SYSTEM_PROCESSOR AMD64)

init_user_prop(HOST_ARCH)
init_user_prop(LLVM_NATIVE_TOOLCHAIN)
init_user_prop(MSVC_BASE)
init_user_prop(WINSDK_BASE)
init_user_prop(WINSDK_VER)

if(NOT HOST_ARCH)
  set(HOST_ARCH x86_64)
endif()
if(HOST_ARCH STREQUAL "aarch64" OR HOST_ARCH STREQUAL "arm64")
  set(TRIPLE_ARCH "aarch64")
  set(WINSDK_ARCH "arm64")
elseif(HOST_ARCH STREQUAL "armv7" OR HOST_ARCH STREQUAL "arm")
  set(TRIPLE_ARCH "armv7")
  set(WINSDK_ARCH "arm")
elseif(HOST_ARCH STREQUAL "i686" OR HOST_ARCH STREQUAL "x86")
  set(TRIPLE_ARCH "i686")
  set(WINSDK_ARCH "x86")
elseif(HOST_ARCH STREQUAL "x86_64" OR HOST_ARCH STREQUAL "x64")
  set(TRIPLE_ARCH "x86_64")
  set(WINSDK_ARCH "x64")
else()
  message(SEND_ERROR "Unknown host architecture ${HOST_ARCH}. Must be aarch64 (or arm64), armv7 (or arm), i686 (or x86), or x86_64 (or x64).")
endif()

set(MSVC_INCLUDE "${MSVC_BASE}/include")
set(ATLMFC_INCLUDE "${MSVC_BASE}/atlmfc/include")
set(MSVC_LIB "${MSVC_BASE}/lib")
set(ATLMFC_LIB "${MSVC_BASE}/atlmfc/lib")
set(WINSDK_INCLUDE "${WINSDK_BASE}/Include/${WINSDK_VER}")
set(WINSDK_LIB "${WINSDK_BASE}/Lib/${WINSDK_VER}")

# Do some sanity checking to make sure we can find a native toolchain and
# that the Windows SDK / MSVC STL directories look kosher.
if(NOT EXISTS "${LLVM_NATIVE_TOOLCHAIN}/bin/clang-cl" OR
   NOT EXISTS "${LLVM_NATIVE_TOOLCHAIN}/bin/lld-link")
  message(SEND_ERROR
          "LLVM_NATIVE_TOOLCHAIN folder '${LLVM_NATIVE_TOOLCHAIN}' does not "
          "point to a valid directory containing bin/clang-cl and bin/lld-link "
          "binaries")
endif()

if(NOT EXISTS "${MSVC_BASE}" OR
   NOT EXISTS "${MSVC_INCLUDE}" OR
   NOT EXISTS "${MSVC_LIB}")
  message(SEND_ERROR
          "CMake variable MSVC_BASE must point to a folder containing MSVC "
          "system headers and libraries")
endif()

if(NOT EXISTS "${WINSDK_BASE}" OR
   NOT EXISTS "${WINSDK_INCLUDE}" OR
   NOT EXISTS "${WINSDK_LIB}")
  message(SEND_ERROR
          "CMake variable WINSDK_BASE and WINSDK_VER must resolve to a valid "
          "Windows SDK installation")
endif()

if(NOT EXISTS "${WINSDK_INCLUDE}/um/Windows.h")
  message(SEND_ERROR "Cannot find Windows.h")
endif()
if(NOT EXISTS "${WINSDK_INCLUDE}/um/WINDOWS.H")
  set(case_sensitive_filesystem TRUE)
endif()

set(CMAKE_C_COMPILER "${LLVM_NATIVE_TOOLCHAIN}/bin/clang-cl" CACHE FILEPATH "")
set(CMAKE_CXX_COMPILER "${LLVM_NATIVE_TOOLCHAIN}/bin/clang-cl" CACHE FILEPATH "")
set(CMAKE_LINKER "${LLVM_NATIVE_TOOLCHAIN}/bin/lld-link" CACHE FILEPATH "")

# Even though we're cross-compiling, we need some native tools (e.g. llvm-tblgen), and those
# native tools have to be built before we can start doing the cross-build.  LLVM supports
# a CROSS_TOOLCHAIN_FLAGS_NATIVE argument which consists of a list of flags to pass to CMake
# when configuring the NATIVE portion of the cross-build.  By default we construct this so
# that it points to the tools in the same location as the native clang-cl that we're using.
list(APPEND _CTF_NATIVE_DEFAULT "-DCMAKE_ASM_COMPILER=${LLVM_NATIVE_TOOLCHAIN}/bin/clang")
list(APPEND _CTF_NATIVE_DEFAULT "-DCMAKE_C_COMPILER=${LLVM_NATIVE_TOOLCHAIN}/bin/clang")
list(APPEND _CTF_NATIVE_DEFAULT "-DCMAKE_CXX_COMPILER=${LLVM_NATIVE_TOOLCHAIN}/bin/clang++")

set(CROSS_TOOLCHAIN_FLAGS_NATIVE "${_CTF_NATIVE_DEFAULT}" CACHE STRING "")

set(COMPILE_FLAGS
    -D_CRT_SECURE_NO_WARNINGS
    --target=${TRIPLE_ARCH}-windows-msvc
    -fms-compatibility-version=19.11
    -imsvc "${ATLMFC_INCLUDE}"
    -imsvc "${MSVC_INCLUDE}"
    -imsvc "${WINSDK_INCLUDE}/ucrt"
    -imsvc "${WINSDK_INCLUDE}/shared"
    -imsvc "${WINSDK_INCLUDE}/um"
    -imsvc "${WINSDK_INCLUDE}/winrt")

if(case_sensitive_filesystem)
  # Ensure all sub-configures use the top-level VFS overlay instead of generating their own.
  init_user_prop(winsdk_vfs_overlay_path)
  if(NOT winsdk_vfs_overlay_path)
    set(winsdk_vfs_overlay_path "${CMAKE_BINARY_DIR}/winsdk_vfs_overlay.yaml")
    generate_winsdk_vfs_overlay("${WINSDK_BASE}/Include/${WINSDK_VER}" "${winsdk_vfs_overlay_path}")
    init_user_prop(winsdk_vfs_overlay_path)
  endif()
  list(APPEND COMPILE_FLAGS
       -Xclang -ivfsoverlay -Xclang "${winsdk_vfs_overlay_path}")
endif()

string(REPLACE ";" " " COMPILE_FLAGS "${COMPILE_FLAGS}")

# We need to preserve any flags that were passed in by the user. However, we
# can't append to CMAKE_C_FLAGS and friends directly, because toolchain files
# will be re-invoked on each reconfigure and therefore need to be idempotent.
# The assignments to the _INITIAL cache variables don't use FORCE, so they'll
# only be populated on the initial configure, and their values won't change
# afterward.
set(_CMAKE_C_FLAGS_INITIAL "${CMAKE_C_FLAGS}" CACHE STRING "")
set(CMAKE_C_FLAGS "${_CMAKE_C_FLAGS_INITIAL} ${COMPILE_FLAGS}" CACHE STRING "" FORCE)

set(_CMAKE_CXX_FLAGS_INITIAL "${CMAKE_CXX_FLAGS}" CACHE STRING "")
set(CMAKE_CXX_FLAGS "${_CMAKE_CXX_FLAGS_INITIAL} ${COMPILE_FLAGS}" CACHE STRING "" FORCE)

set(LINK_FLAGS
    # Prevent CMake from attempting to invoke mt.exe. It only recognizes the slashed form and not the dashed form.
    /manifest:no

    -libpath:"${ATLMFC_LIB}/${WINSDK_ARCH}"
    -libpath:"${MSVC_LIB}/${WINSDK_ARCH}"
    -libpath:"${WINSDK_LIB}/ucrt/${WINSDK_ARCH}"
    -libpath:"${WINSDK_LIB}/um/${WINSDK_ARCH}")

if(case_sensitive_filesystem)
  # Ensure all sub-configures use the top-level symlinks dir instead of generating their own.
  init_user_prop(winsdk_lib_symlinks_dir)
  if(NOT winsdk_lib_symlinks_dir)
    set(winsdk_lib_symlinks_dir "${CMAKE_BINARY_DIR}/winsdk_lib_symlinks")
    generate_winsdk_lib_symlinks("${WINSDK_BASE}/Lib/${WINSDK_VER}/um/${WINSDK_ARCH}" "${winsdk_lib_symlinks_dir}")
    init_user_prop(winsdk_lib_symlinks_dir)
  endif()
  list(APPEND LINK_FLAGS
       -libpath:"${winsdk_lib_symlinks_dir}")
endif()

string(REPLACE ";" " " LINK_FLAGS "${LINK_FLAGS}")

# See explanation for compiler flags above for the _INITIAL variables.
set(_CMAKE_EXE_LINKER_FLAGS_INITIAL "${CMAKE_EXE_LINKER_FLAGS}" CACHE STRING "")
set(CMAKE_EXE_LINKER_FLAGS "${_CMAKE_EXE_LINKER_FLAGS_INITIAL} ${LINK_FLAGS}" CACHE STRING "" FORCE)

set(_CMAKE_MODULE_LINKER_FLAGS_INITIAL "${CMAKE_MODULE_LINKER_FLAGS}" CACHE STRING "")
set(CMAKE_MODULE_LINKER_FLAGS "${_CMAKE_MODULE_LINKER_FLAGS_INITIAL} ${LINK_FLAGS}" CACHE STRING "" FORCE)

set(_CMAKE_SHARED_LINKER_FLAGS_INITIAL "${CMAKE_SHARED_LINKER_FLAGS}" CACHE STRING "")
set(CMAKE_SHARED_LINKER_FLAGS "${_CMAKE_SHARED_LINKER_FLAGS_INITIAL} ${LINK_FLAGS}" CACHE STRING "" FORCE)

# CMake populates these with a bunch of unnecessary libraries, which requires
# extra case-correcting symlinks and what not. Instead, let projects explicitly
# control which libraries they require.
set(CMAKE_C_STANDARD_LIBRARIES "" CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD_LIBRARIES "" CACHE STRING "" FORCE)

# Allow clang-cl to work with macOS paths.
set(CMAKE_USER_MAKE_RULES_OVERRIDE "${CMAKE_CURRENT_LIST_DIR}/ClangClCMakeCompileRules.cmake")
