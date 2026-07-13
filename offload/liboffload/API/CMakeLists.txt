# We want to clang-format the generated files if possible, since OffloadAPI.h is
# the main public header for liboffload. Generate them in a temporary location,
# then clang-format and copy them to the proper location. If clang-format is
# missing just copy them.
# Ideally we'd just clang-format them in place and avoid the copy but cmake
# gets confused about the same path being a byproduct of two custom commands.

set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/OffloadAPI.td)
set(files_to_copy "")
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)

macro(offload_tablegen file)
    tablegen(OFFLOAD generated/${file}.gen ${ARGN})
    list(APPEND files_to_copy ${file})
endmacro()

offload_tablegen(OffloadAPI.h -gen-api)
offload_tablegen(OffloadEntryPoints.inc -gen-entry-points)
offload_tablegen(OffloadFuncs.inc -gen-func-names)
offload_tablegen(OffloadImplFuncDecls.inc -gen-impl-func-decls)
offload_tablegen(OffloadPrint.hpp -gen-print-header)

add_public_tablegen_target(OffloadGenerate)

add_custom_target(OffloadAPI DEPENDS OffloadGenerate)
find_program(clang_format clang-format PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
if (clang_format)
    foreach(file IN LISTS files_to_copy)
      add_custom_command(
          OUTPUT ${file}
          COMMAND ${clang_format} -i generated/${file}.gen
          COMMAND ${CMAKE_COMMAND} -E copy_if_different generated/${file}.gen ${CMAKE_CURRENT_BINARY_DIR}/${file}
          DEPENDS generated/${file}.gen
      )
      add_custom_target(OffloadAPI.${file} DEPENDS ${file})
      add_dependencies(OffloadAPI OffloadAPI.${file})
    endforeach()
else()
    message(WARNING "clang-format not found, the generated Offload API headers will not be formatted")
    foreach(file IN LISTS files_to_copy)
        add_custom_command(
            OUTPUT ${file}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different generated/${file}.gen ${CMAKE_CURRENT_BINARY_DIR}/${file}
            DEPENDS generated/${file}.gen
        )
        add_custom_target(OffloadAPI.${file} DEPENDS ${file})
        add_dependencies(OffloadAPI OffloadAPI.${file})
    endforeach()
endif()
