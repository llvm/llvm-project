; Test to check the callgraph in summary
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; RUN: llvm-dis -o - %t.o | FileCheck %s --check-prefix=DIS
; XFAIL: *

@ifunc = dso_local ifunc i32 (), bitcast (i32 ()* ()* @resolver to i32 ()*)

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 ()* @resolver() #0 {
  ret i32 ()* @called
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @called() #0 {
  ret i32 1
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
  %1 = call i32 @ifunc()
  ret i32 0
}

; CHECK:       <GLOBALVAL_SUMMARY_BLOCK
; CHECK-NEXT:    <VERSION
; CHECK-NEXT:    <FLAGS
; CHECK-NEXT:    <PERMODULE {{.*}} op0=[[RESOLVERID:[0-9]+]] {{.*}} op7=[[CALLEDID:[0-9]+]]/>
; CHECK-NEXT:    <PERMODULE {{.*}} op0=[[CALLEDID]]
; CHECK-NEXT:    <PERMODULE {{.*}} op7=[[IFUNCID:[0-9]+]]/>
; CHECK-NEXT:    <IFUNC {{.*}} op0=[[IFUNCID]] {{.*}} op2=[[RESOLVERID]]/>
; CHECK-NEXT:    <BLOCK_COUNT op0=3/>
; CHECK-NEXT:  </GLOBALVAL_SUMMARY_BLOCK>

; DIS: ^0 = module: (path: "{{.*}}", hash: (0, 0, 0, 0, 0))
; DIS: ^1 = gv: (name: "ifunc", summaries: (ifunc: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0), resolver: ^4))) ; guid = 1234216394087659437
; DIS: ^2 = gv: (name: "called", summaries: (function: (module: ^0, flags: (linkage: internal, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0), insts: 1))) ; guid = 4806741020937274681
; DIS: ^3 = gv: (name: "main", summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0), insts: 2, calls: ((callee: ^1))))) ; guid = 15822663052811949562
; DIS: ^4 = gv: (name: "resolver", summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0), insts: 1, refs: (^2)))) ; guid = 18291748799076262136
; DIS: ^5 = blockcount: 3
