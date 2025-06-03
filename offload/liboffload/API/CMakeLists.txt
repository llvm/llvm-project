# The OffloadGenerate target is used to regenerate the generated files in the
# include directory. These files are checked in with the rest of the source,
# therefore it is only needed when making changes to the API.

find_program(clang_format clang-format PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
if (clang_format)
    set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/OffloadAPI.td)

    tablegen(OFFLOAD OffloadAPI.h -gen-api)
    tablegen(OFFLOAD OffloadEntryPoints.inc -gen-entry-points)
    tablegen(OFFLOAD OffloadFuncs.inc -gen-func-names)
    tablegen(OFFLOAD OffloadImplFuncDecls.inc -gen-impl-func-decls)
    tablegen(OFFLOAD OffloadPrint.hpp -gen-print-header)
    tablegen(OFFLOAD OffloadErrcodes.inc -gen-errcodes)

    set(files_to_copy "OffloadAPI.h;OffloadEntryPoints.inc;OffloadFuncs.inc;OffloadImplFuncDecls.inc;OffloadPrint.hpp")
    set(generated_dir ${CMAKE_CURRENT_SOURCE_DIR}/../include/generated)
    set(shared_dir ${LIBOMPTARGET_INCLUDE_DIR}/Shared)
    add_public_tablegen_target(OffloadGenerate)

    add_custom_target(OffloadAPI DEPENDS OffloadGenerate)
      add_custom_command(
          OUTPUT ${shared_dir}/OffloadErrcodes.inc
          COMMAND ${clang_format} -i OffloadErrcodes.inc
          COMMAND ${CMAKE_COMMAND} -E copy_if_different OffloadErrcodes.inc ${shared_dir}
          DEPENDS OffloadErrcodes.inc
      )
      add_custom_target(OffloadAPI.OffloadErrcodes.inc DEPENDS ${shared_dir}/OffloadErrcodes.inc)
      add_dependencies(OffloadAPI OffloadAPI.OffloadErrcodes.inc)
    foreach(file IN LISTS files_to_copy)
      add_custom_command(
          OUTPUT ${generated_dir}/${file}
          COMMAND ${clang_format} -i ${file}
          COMMAND ${CMAKE_COMMAND} -E make_directory ${generated_dir}
          COMMAND ${CMAKE_COMMAND} -E copy_if_different ${file} ${generated_dir}
          DEPENDS ${file}
      )
      add_custom_target(OffloadAPI.${file} DEPENDS ${generated_dir}/${file})
      add_dependencies(OffloadAPI OffloadAPI.${file})
    endforeach()
else()
    message(WARNING "clang-format was not found, so the OffloadGenerate target\
        will not be available. Offload will still build, but you will not be\
        able to make changes to the API.")
endif()
