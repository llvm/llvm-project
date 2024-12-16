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

    set(OFFLOAD_GENERATED_FILES ${TABLEGEN_OUTPUT})
    add_public_tablegen_target(OffloadGenerate)
    add_custom_command(TARGET OffloadGenerate POST_BUILD COMMAND ${CLANG_FORMAT}
        -i ${OFFLOAD_GENERATED_FILES})
    add_custom_command(TARGET OffloadGenerate POST_BUILD COMMAND ${CMAKE_COMMAND}
        -E copy_if_different ${OFFLOAD_GENERATED_FILES} "${CMAKE_CURRENT_SOURCE_DIR}/../include/generated")
else()
    message(WARNING "clang-format was not found, so the OffloadGenerate target\
        will not be available. Offload will still build, but you will not be\
        able to make changes to the API.")
endif()
