# The OffloadGenerate target is used to regenerate the generated files in the
# include directory. These files are checked in with the rest of the source,
# therefore it is only needed when making changes to the API.

find_program(CLANG_FORMAT clang-format PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
if (CLANG_FORMAT)
    set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/OffloadAPI.td)

    tablegen(OFFLOAD OffloadAPI.h -gen-api)
    tablegen(OFFLOAD OffloadEntryPoints.inc -gen-entry-points)
    tablegen(OFFLOAD OffloadFuncs.inc -gen-func-names)
    tablegen(OFFLOAD OffloadImplFuncDecls.inc -gen-impl-func-decls)
    tablegen(OFFLOAD OffloadPrint.hpp -gen-print-header)
    tablegen(OFFLOAD OffloadErrcodes.inc -gen-errcodes)

    set(FILES_TO_COPY "OffloadAPI.h;OffloadEntryPoints.inc;OffloadFuncs.inc;OffloadImplFuncDecls.inc;OffloadPrint.hpp")
    set(GEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include/generated)
    add_public_tablegen_target(OffloadGenerate)
    add_custom_command(TARGET OffloadGenerate POST_BUILD COMMAND ${CLANG_FORMAT}
        -i ${TABLEGEN_OUTPUT})
    add_custom_command(TARGET OffloadGenerate POST_BUILD COMMAND ${CMAKE_COMMAND}
        -E copy_if_different ${FILES_TO_COPY} ${GEN_DIR})
    add_custom_command(TARGET OffloadGenerate POST_BUILD COMMAND ${CMAKE_COMMAND}
        -E copy_if_different OffloadErrcodes.inc "${LIBOMPTARGET_INCLUDE_DIR}/Shared/OffloadErrcodes.inc")
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
