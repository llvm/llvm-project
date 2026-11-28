include(TableGen)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/AccAPI.td)

offload_tablegen(AccFModule.inc -gen-acc-f-module)
offload_tablegen(openacc/AccCEnums.inc -gen-acc-c-enums)
offload_tablegen(acc.exports -gen-acc-exports)
offload_tablegen(AccPrintEnums.inc -gen-acc-print-enums)
offload_tablegen(AccFEntries.inc -gen-acc-f-entries)
offload_tablegen(AccFImplDecls.inc -gen-acc-f-impl-decls)

add_public_tablegen_target(AccAPITableGen)
