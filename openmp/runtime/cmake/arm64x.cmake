# check if we have the right linker flags to enable Arm64X
include(CheckLinkerFlag)
check_linker_flag(CXX "LINKER:/linkreprofullpathrsp:test_rsp" LIBOMP_HAVE_LINKREPROFULLPATHRSP_FLAG)
if(NOT LIBOMP_HAVE_LINKREPROFULLPATHRSP_FLAG)
  message(FATAL_ERROR "Arm64X builds are enabled but the linker does not support the required flag. "
    "Either update Visual Studio if using link.exe or add lld to LLVM_ENABLE_PROJECTS to use a newer lld.")
endif()

# directory where the link.rsp file generated during arm64 build will be stored
set(arm64ReproDir "${LLVM_BINARY_DIR}/runtimes/repros-arm64ec")

# Don't install the runtime if we are doing an arm64ec build for arm64x.
# The hybrid arm64x runtime will get installed by the host (default) runtime build
if (LIBOMP_ARCH STREQUAL "arm64ec")
  set(CMAKE_SKIP_INSTALL_RULES On)
endif()

# This function reads in the content of the rsp file outputted from the arm64ec build for a target,
# then passes the arm64ec libs and objs to the linker using /machine:arm64x to combine them with the
# arm64 counterparts and create an arm64x binary.
function(set_arm64ec_dll_dependencies target)
  set(REPRO_FILE "${arm64ReproDir}/${target}.rsp")
  file(STRINGS "${REPRO_FILE}" ARM64_OBJS REGEX obj\"$)
  file(STRINGS "${REPRO_FILE}" ARM64_LIBS REGEX lib\"$)
  string(REPLACE "\"" ";" ARM64_OBJS "${ARM64_OBJS}")
  string(REPLACE "\"" ";" ARM64_LIBS "${ARM64_LIBS}")

  get_target_property(libs "${target}" LINK_FLAGS)
  set(non_def "")

  # Separate out the /def flag from the other link flags, so we can replcae it with /defArm64Native.
  foreach(lib ${libs})
    if(lib MATCHES ".*\.def")
      string(REPLACE "/DEF:" "" def ${lib})
    else()
      list(APPEND non_def "${lib}")
    endif()
  endforeach()
  # Remove the /def link flag
  set_target_properties("${target}" PROPERTIES LINK_FLAGS "${non_def}")

  target_sources("${target}" PRIVATE ${ARM64_OBJS})
  target_link_options("${target}" PRIVATE /machine:arm64x "/def:${arm64ReproDir}/${target}.def" "/defArm64Native:${def}")
endfunction()

# Replace the /def flag with /defArm64Native and add the arm64ec /def file.
function(set_arm64ec_lib_dependencies target)
  get_target_property(opts ${target} STATIC_LIBRARY_OPTIONS)
  string(REPLACE "/DEF:" "/defArm64Native:" opts "${opts}")
  set_target_properties(${target} PROPERTIES STATIC_LIBRARY_OPTIONS "/machine:arm64x;${opts};/def:${arm64ReproDir}/${target}.def")
endfunction()

# Copy the def file for arm64ec to the repros directory so we can use it in the arm64x builds and add the linkreprofullpathrsp flag.
function(handle_arm64ec_target target)
  get_target_property(type "${target}" TYPE)
  if(type STREQUAL "SHARED_LIBRARY")
    get_target_property(libs "${target}" LINK_FLAGS)
  elseif(type STREQUAL "STATIC_LIBRARY")
    get_target_property(libs "${target}" STATIC_LIBRARY_OPTIONS)
  endif()
  list(FILTER libs INCLUDE REGEX ".*\.def")
  string(REPLACE "/DEF:" "" def "${libs}")

  add_custom_target("${target}.def"
                         BYPRODUCTS "${arm64ReproDir}/${target}.def"
                         COMMAND ${CMAKE_COMMAND} -E copy
                         "${def}"
                         "${arm64ReproDir}/${target}.def"
                         DEPENDS "${def}")
  add_dependencies(${target} "${target}.def")
  # tell the linker to produce this special rsp file that has absolute paths to its inputs
  if(type STREQUAL "SHARED_LIBRARY")
    target_link_options(${target} PRIVATE "/LINKREPROFULLPATHRSP:${arm64ReproDir}/${target}.rsp")
  endif()
endfunction()

# Handle the targets we have requested arm64x builds for.
function(handle_arm64x)
  # During the arm64ec build, create rsp files that containes the absolute path to the inputs passed to the linker (objs, libs).
  if("${LIBOMP_ARCH}" STREQUAL "arm64ec")
    file(MAKE_DIRECTORY ${arm64ReproDir})
    foreach (target ${ARM64X_TARGETS})
      handle_arm64ec_target("${target}")
    endforeach()

  # During the ARM64 build, modify the link step appropriately to produce an arm64x binary
  elseif("${LIBOMP_ARCH}" STREQUAL "aarch64")
    foreach (target ${ARM64X_TARGETS})
      get_target_property(type ${target} TYPE)
      if(type STREQUAL "SHARED_LIBRARY")
        set_arm64ec_dll_dependencies("${target}")
      elseif(type STREQUAL "STATIC_LIBRARY")
        set_arm64ec_lib_dependencies("${target}")
      endif()
    endforeach()
  endif()
endfunction()
