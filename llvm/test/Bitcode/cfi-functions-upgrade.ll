; RUN: llvm-bcanalyzer -dump %S/Inputs/cfi-summary-upgrade.bc | FileCheck %s --check-prefix=OLD-SUMMARY
; RUN: llvm-dis < %S/Inputs/cfi-summary-upgrade.bc | FileCheck %s --check-prefix=UPGRADED-SUMMARY

; RUN: llvm-bcanalyzer -dump %S/Inputs/cfi-functions-upgrade.bc | FileCheck %s --check-prefix=OLD-METADATA
; RUN: llvm-dis < %S/Inputs/cfi-functions-upgrade.bc | llvm-as | llvm-bcanalyzer -dump | FileCheck %s --check-prefix=UPGRADED-METADATA

; OLD-SUMMARY: <GLOBALVAL_SUMMARY_BLOCK
; OLD-SUMMARY: <VERSION op0=13/>
; OLD-SUMMARY: <CFI_FUNCTION_DEFS op0=0 op1=3/>
; OLD-SUMMARY: <CFI_FUNCTION_DECLS op0=3 op1=3/>

; This just tests that the old summary got loaded successfully. llvm-dis does't
; have a representation for the 2 cfi tables.
; UPGRADED-SUMMARY: ^0 = module:
; UPGRADED-SUMMARY: ^1 = gv: (guid: 6699318081062747564,
; UPGRADED-SUMMARY: ^2 = gv: (guid: 14771895114995649345,
; UPGRADED-SUMMARY: ^3 = typeid: (name: "typeid1",

; This tests a roundtrip of old metadata -> new metadata.
; OLD-METADATA: <METADATA_BLOCK
; OLD-METADATA: <NODE op0={{[0-9]+}} op1={{[0-9]+}} op2={{[0-9]+}}/>

; UPGRADED-METADATA: <METADATA_BLOCK
; UPGRADED-METADATA: <NODE op0={{[0-9]+}} op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}}/>
