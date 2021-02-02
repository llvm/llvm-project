; Test to check the callgraph in summary
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; RUN: opt -module-summary %p/Inputs/thinlto-alias.ll -o %t2.o
; RUN: llvm-dis -o - %t2.o | FileCheck %s --check-prefix=DIS
; Round trip it through llvm-as
; RUN: llvm-dis -o - %t2.o | llvm-as -o - | llvm-dis -o - | FileCheck %s --check-prefix=DIS
; RUN: llvm-lto -thinlto -o %t3 %t.o %t2.o
; RUN: llvm-bcanalyzer -dump %t3.thinlto.bc | FileCheck %s --check-prefix=COMBINED
; RUN: llvm-dis -o - %t3.thinlto.bc | FileCheck %s --check-prefix=COMBINED-DIS
; Round trip it through llvm-as
; RUN: llvm-dis -o - %t3.thinlto.bc | llvm-as -o - | llvm-dis -o - | FileCheck %s --check-prefix=COMBINED-DIS

; CHECK: <SOURCE_FILENAME
; "main"
; CHECK-NEXT: <FUNCTION op0=0 op1=4
; "analias"
; CHECK-NEXT: <FUNCTION op0=4 op1=7
; CHECK:       <GLOBALVAL_SUMMARY_BLOCK
; CHECK-NEXT:    <VERSION
; CHECK-NEXT:    <FLAGS
; See if the call to func is registered.
; The value id 1 matches the second FUNCTION record above.
; CHECK-NEXT:    <PERMODULE {{.*}} op7=1/>
; CHECK-NEXT:    <BLOCK_COUNT op0=1/>
; CHECK-NEXT:  </GLOBALVAL_SUMMARY_BLOCK>

; CHECK: <STRTAB_BLOCK
; CHECK-NEXT: blob data = 'mainanalias{{.*}}'

; COMBINED:       <GLOBALVAL_SUMMARY_BLOCK
; COMBINED-NEXT:    <VERSION
; COMBINED-NEXT:    <FLAGS
; See if the call to analias is registered, using the expected value id.
; COMBINED-NEXT:    <VALUE_GUID op0=[[ALIASID:[0-9]+]] op1=-5751648690987223394/>
; COMBINED-NEXT:    <VALUE_GUID
; COMBINED-NEXT:    <VALUE_GUID op0=[[ALIASEEID:[0-9]+]] op1=-1039159065113703048/>
; COMBINED-NEXT:    <COMBINED {{.*}} op9=[[ALIASID]]/>
; COMBINED-NEXT:    <COMBINED {{.*}}
; COMBINED-NEXT:    <COMBINED_ALIAS  {{.*}} op3=[[ALIASEEID]]
; COMBINED-NEXT:    <BLOCK_COUNT op0=2/>
; COMBINED-NEXT:  </GLOBALVAL_SUMMARY_BLOCK

; ModuleID = 'thinlto-function-summary-callgraph.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @main() {
entry:
    call void (...) @analias()
    ret i32 0
}

declare void @analias(...)

; DIS: ^0 = module: (path: "{{.*}}", hash: (0, 0, 0, 0, 0))
; DIS: ^1 = gv: (name: "analias", summaries: (alias: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), aliasee: ^2))) ; guid = 12695095382722328222
; DIS: ^2 = gv: (name: "aliasee", summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), insts: 1))) ; guid = 17407585008595848568
; DIS: ^3 = blockcount: 1

; COMBINED-DIS: ^0 = module: (path: "{{.*}}thinlto-alias.ll.tmp.o", hash: (0, 0, 0, 0, 0))
; COMBINED-DIS: ^1 = module: (path: "{{.*}}thinlto-alias.ll.tmp2.o", hash: (0, 0, 0, 0, 0))
; COMBINED-DIS: ^2 = gv: (guid: 12695095382722328222, summaries: (alias: (module: ^1, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), aliasee: ^4)))
; COMBINED-DIS: ^3 = gv: (guid: 15822663052811949562, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), insts: 2, calls: ((callee: ^2)))))
; COMBINED-DIS: ^4 = gv: (guid: 17407585008595848568, summaries: (function: (module: ^1, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), insts: 1)))
; COMBINED-DIS: ^5 = blockcount: 2
