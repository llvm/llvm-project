; Test to check both the callgraph and refgraph in summary
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; RUN: llvm-dis -o - %t.o | FileCheck %s --check-prefix=DIS
; Round trip it through llvm-as
; RUN: llvm-dis -o - %t.o | llvm-as -o - | llvm-dis -o - | FileCheck %s --check-prefix=DIS

; CHECK: <SOURCE_FILENAME
; "bar"
; CHECK-NEXT: <GLOBALVAR {{.*}} op0=0 op1=3
; "globalvar"
; CHECK-NEXT: <GLOBALVAR {{.*}} op0=3 op1=9
; "func"
; CHECK-NEXT: <FUNCTION op0=12 op1=4
; "func2"
; CHECK-NEXT: <FUNCTION op0=16 op1=5
; "foo"
; CHECK-NEXT: <FUNCTION op0=21 op1=3
; "func3"
; CHECK-NEXT: <FUNCTION op0=24 op1=5
; "W"
; CHECK-NEXT: <FUNCTION op0=29 op1=1
; "X"
; CHECK-NEXT: <FUNCTION op0=30 op1=1
; "Y"
; CHECK-NEXT: <FUNCTION op0=31 op1=1
; "Z"
; CHECK-NEXT: <FUNCTION op0=32 op1=1
; "llvm.ctpop.i8"
; CHECK-NEXT: <FUNCTION op0=33 op1=13
; "main"
; CHECK-NEXT: <FUNCTION op0=46 op1=4

; See if the calls and other references are recorded properly using the
; expected value id and other information as appropriate (callsite cout
; for calls). Use different linkage types for the various test cases to
; distinguish the test cases here (op1 contains the linkage type).
; Note that op3 contains the # non-call references.
; This also ensures that we didn't include a call or reference to intrinsic
; llvm.ctpop.i8.
; CHECK:       <GLOBALVAL_SUMMARY_BLOCK
; Function main contains call to func, as well as address reference to func:
; op0=main op4=func op5=func
; CHECK-DAG:    <PERMODULE {{.*}} op0=11 op1=0 {{.*}} op4=1 op5=0 op6=2 op7=2/>
; Function W contains a call to func3 as well as a reference to globalvar:
; op0=W op4=globalvar op5=func3
; CHECK-DAG:    <PERMODULE {{.*}} op0=6 op1=5 {{.*}} op4=1 op5=0 op6=1 op7=5/>
; Function X contains call to foo, as well as address reference to foo
; which is in the same instruction as the call:
; op0=X op4=foo op5=foo
; CHECK-DAG:    <PERMODULE {{.*}} op0=7 op1=1 {{.*}} op4=1 op5=0 op6=4 op7=4/>
; Function Y contains call to func2, and ensures we don't incorrectly add
; a reference to it when reached while earlier analyzing the phi using its
; return value:
; op0=Y op4=func2
; CHECK-DAG:    <PERMODULE {{.*}} op0=8 op1=72 {{.*}} op4=0 op5=0 op6=3/>
; Function Z contains call to func2, and ensures we don't incorrectly add
; a reference to it when reached while analyzing subsequent use of its return
; value:
; op0=Z op4=func2
; CHECK-DAG:    <PERMODULE {{.*}} op0=9 op1=3 {{.*}} op4=0 op5=0 op6=3/>
; Variable bar initialization contains address reference to func:
; op0=bar op2=func
; CHECK-DAG:    <PERMODULE_GLOBALVAR_INIT_REFS {{.*}} op0=0 op1=0 op2=1 op3=2/>
; CHECK:  </GLOBALVAL_SUMMARY_BLOCK>

; CHECK: <STRTAB_BLOCK
; CHECK-NEXT: blob data = 'barglobalvarfuncfunc2foofunc3WXYZllvm.ctpop.i8main{{.*}}'

; ModuleID = 'thinlto-function-summary-refgraph.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@bar = global void (...)* bitcast (void ()* @func to void (...)*), align 8

@globalvar = global i32 0, align 4

declare void @func() #0
declare i32 @func2(...) #1
declare void @foo(i8* %F) #0
declare i32 @func3(i32* dereferenceable(4)) #2

; Function Attrs: nounwind uwtable
define weak_odr void @W() #0 {
entry:
  %call = tail call i32 @func3(i32* nonnull dereferenceable(4) @globalvar)
  ret void
}

; Function Attrs: nounwind uwtable
define available_externally void @X() #0 {
entry:
  call void @foo(i8* bitcast (void (i8*)* @foo to i8*))
  ret void
}

; Function Attrs: nounwind uwtable
define private i32 @Y(i32 %i) #0 {
entry:
  %cmp3 = icmp slt i32 %i, 10
  br i1 %cmp3, label %while.body.preheader, label %while.end

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %j.05 = phi i32 [ %add, %while.body ], [ 0, %while.body.preheader ]
  %i.addr.04 = phi i32 [ %inc, %while.body ], [ %i, %while.body.preheader ]
  %inc = add nsw i32 %i.addr.04, 1
  %call = tail call i32 (...) @func2() #2
  %add = add nsw i32 %call, %j.05
  %exitcond = icmp eq i32 %inc, 10
  br i1 %exitcond, label %while.end.loopexit, label %while.body

while.end.loopexit:                               ; preds = %while.body
  %add.lcssa = phi i32 [ %add, %while.body ]
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %j.0.lcssa = phi i32 [ 0, %entry ], [ %add.lcssa, %while.end.loopexit ]
  ret i32 %j.0.lcssa
}

; Function Attrs: nounwind uwtable
define linkonce_odr i32 @Z() #0 {
entry:
  %call = tail call i32 (...) @func2() #2
  ret i32 %call
}

declare i8 @llvm.ctpop.i8(i8)

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %foo = alloca void (...)*, align 8
  store i32 0, i32* %retval, align 4
  store void (...)* bitcast (void ()* @func to void (...)*), void (...)** %foo, align 8
  %0 = load void (...)*, void (...)** %foo, align 8
  call void (...) %0()
  call void @func()
  call i8  @llvm.ctpop.i8( i8 10 )
  ret i32 0
}

; Don't try to match summary IDs. The numbering depends on the map iteration
; order, which depends on GUID, and the private function Y GUID will depend
; on the path to the test.
; DIS: ^0 = module: (path: "{{.*}}", hash: (0, 0, 0, 0, 0))
; DIS-DAG: = gv: (name: "Z", summaries: (function: (module: ^0, flags: (linkage: linkonce_odr, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), insts: 2, calls: ((callee: ^{{.*}}))))) ; guid = 104084381700047393
; DIS-DAG: = gv: (name: "X", summaries: (function: (module: ^0, flags: (linkage: available_externally, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), insts: 2, calls: ((callee: ^{{.*}})), refs: (^{{.*}})))) ; guid = 1881667236089500162
; DIS-DAG: = gv: (name: "W", summaries: (function: (module: ^0, flags: (linkage: weak_odr, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), insts: 2, calls: ((callee: ^{{.*}})), refs: (^{{.*}})))) ; guid = 5790125716599269729
; DIS-DAG: = gv: (name: "foo") ; guid = 6699318081062747564
; DIS-DAG: = gv: (name: "func") ; guid = 7289175272376759421
; DIS-DAG: = gv: (name: "func3") ; guid = 11517462787082255043
; DIS-DAG: = gv: (name: "globalvar", summaries: (variable: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), varFlags: (readonly: 1)))) ; guid = 12887606300320728018
; DIS-DAG: = gv: (name: "func2") ; guid = 14069196320850861797
; DIS-DAG: = gv: (name: "llvm.ctpop.i8") ; guid = 15254915475081819833
; DIS-DAG: = gv: (name: "main", summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), insts: 9, calls: ((callee: ^{{.*}})), refs: (^{{.*}})))) ; guid = 15822663052811949562
; DIS-DAG: = gv: (name: "bar", summaries: (variable: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), varFlags: (readonly: 1),  refs: (^{{.*}})))) ; guid = 16434608426314478903
; Don't try to match the exact GUID. Since it is private, the file path
; will get hashed, and that will be test dependent.
; DIS-DAG: = gv: (name: "Y", summaries: (function: (module: ^0, flags: (linkage: private, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0), insts: 14, calls: ((callee: ^{{.*}}))))) ; guid =
