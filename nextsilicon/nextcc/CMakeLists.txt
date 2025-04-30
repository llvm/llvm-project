cmake_minimum_required(VERSION 3.4.3)
project(nextcc)

option(LLVM_DIR_NAME "Relative path of LLVM installation")

set(NEXTCC_SECTION_NAME ".nextcc")
set(NEXTCC_VERSION_MAJOR 3)
set(NEXTCC_VERSION_MINOR 0)
set(NEXTCC_VERSION_PATCHLEVEL 0)
set(NEXTCC_VERSION "${NEXTCC_VERSION_MAJOR}.${NEXTCC_VERSION_MINOR}.${NEXTCC_VERSION_PATCHLEVEL}")

set(NEXTCC_FILENAME "nextcc-script")

# Adding a configuration file with the toolchain version. This version helps developers
# distinguish between versions when our internal interface between the compiler and
# nextutils breaks.
set(NEXT_TOOLCHAIN_VERSION_INPUT_FILENAME "next_toolchain_version.txt.in")
set(NEXT_TOOLCHAIN_VERSION_OUTPUT_FILENAME "next_toolchain_version.txt")
set(NEXT_TOOLCHAIN_VERSION_MAJOR 5)
set(NEXT_TOOLCHAIN_VERSION_MINOR 19)
set(NEXT_TOOLCHAIN_VERSION_PATCHLEVEL 1)
set(NEXT_TOOLCHAIN_VERSION "${NEXT_TOOLCHAIN_VERSION_MAJOR}.${NEXT_TOOLCHAIN_VERSION_MINOR}.${NEXT_TOOLCHAIN_VERSION_PATCHLEVEL}")

configure_file(
  ${NEXT_TOOLCHAIN_VERSION_INPUT_FILENAME}
  ${NEXT_TOOLCHAIN_VERSION_OUTPUT_FILENAME}
)
install(FILES ${CMAKE_BINARY_DIR}/${NEXT_TOOLCHAIN_VERSION_OUTPUT_FILENAME} DESTINATION bin)

configure_file(nextcc.in ${NEXTCC_FILENAME} @ONLY)
install(PROGRAMS ${CMAKE_BINARY_DIR}/${NEXTCC_FILENAME} DESTINATION bin)

macro(add_nextcc_symlink name)
  set(target_name nextcc-symlink-${name})
  set(output_path ${CMAKE_BINARY_DIR}/${name})
  add_custom_target(${target_name} ALL
    DEPENDS ${output_path}
  )
  add_custom_command(
    OUTPUT ${output_path}
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${NEXTCC_FILENAME} ${output_path}
  )
  install(PROGRAMS ${output_path} DESTINATION bin)
endmacro()

set(NEXTCC_LINKS_TO_CREATE
	nextcc nextcxx nextfort nextflang
	flatcc flatcxx flatfort flatflang
)
foreach(link ${NEXTCC_LINKS_TO_CREATE})
  add_nextcc_symlink(${link})
endforeach()

configure_file(version.h.in include/nextcc/version.h)
install(
  FILES ${CMAKE_BINARY_DIR}/include/nextcc/version.h
  DESTINATION include/nextcc
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
)

function(add_bitcode_object name)
  set(BINARY_PATH ${CMAKE_BINARY_DIR}/${name}.o)
  set(SOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${name}.c)

  add_custom_target(${name} ALL
    DEPENDS ${BINARY_PATH}
  )
  add_custom_command(
    OUTPUT ${BINARY_PATH}
    COMMAND ${CMAKE_BINARY_DIR}/nextcc -fPIC -I${CMAKE_BINARY_DIR}/include
            -c ${SOURCE_PATH} --output=${BINARY_PATH} ${ARGN}
    DEPENDS ${SOURCE_PATH} nextcc-symlink-nextcc
  )
  install(PROGRAMS ${BINARY_PATH} DESTINATION bin)
endfunction()

# By making a bitcode module we partially avoid depending on the default target.
add_bitcode_object(nextcc_version)
add_bitcode_object(nextloader)

configure_file(nextloader.script nextloader.script @ONLY)
install(FILES nextloader.script DESTINATION bin)
