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
    message(WARNING "clang-format was not found, so the OffloadGenerate target\
        will not be available. Offload will still build, but you will not be\
        able to make changes to the API.")
endif()
