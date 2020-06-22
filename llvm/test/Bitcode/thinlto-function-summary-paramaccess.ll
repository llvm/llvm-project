; REQUIRES: aarch64-registered-target

; For convenience, to show what is being serialized.
; RUN: opt -S -passes="print<stack-safety-local>" -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=SSI

; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-bcanalyzer -dump %t.bc | FileCheck %s -check-prefixes=BC

; RUN: llvm-dis -o - %t.bc | FileCheck %s --check-prefix=DIS
; Round trip it through llvm-as
; RUN: llvm-dis -o - %t.bc | llvm-as -o - | llvm-dis -o - | FileCheck %s --check-prefix=DIS

; RUN: opt -thinlto-bc %s -o %t.bc
; RUN: llvm-bcanalyzer -dump %t.bc | FileCheck %s -check-prefixes=BC

; RUN: llvm-dis -o - %t.bc | FileCheck %s --check-prefix=DIS
; Round trip it through llvm-as
; RUN: llvm-dis -o - %t.bc | llvm-as -o - | llvm-dis -o - | FileCheck %s --check-prefix=DIS

; DIS: ^0 = module: (path: "{{.*}}", hash: ({{.*}}))
; ModuleID = 'thinlto-function-summary-paramaccess.ll'
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux"

attributes #0 = { noinline sanitize_memtag "target-features"="+mte,+neon" }

; BC-LABEL: <GLOBALVAL_SUMMARY_BLOCK
; BC-NEXT: <VERSION
; BC-NEXT: <FLAGS

; DIS-DAG: = gv: (name: "Callee") ; guid = 900789920918863816
declare void @Callee(i8* %p)

; DIS-DAG: = gv: (name: "Callee2") ; guid = 72710208629861106
declare void @Callee2(i32 %x, i8* %p)

; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "NoParam", summaries: {{.*}} guid = 10287433468618421703
define void @NoParam() #0 {
entry:
  ret void
}

; SSI-LABEL: function 'IntParam'
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "IntParam", summaries: {{.*}} guid = 13164714711077064397
define void @IntParam(i32 %x) #0 {
entry:
  ret void
}

; SSI-LABEL: for function 'WriteNone'
; SSI: p[]: empty-set
; BC-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=0 op3=0/>
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "WriteNone", summaries: {{.*}} params: ((param: 0, offset: [0, -1]))))) ; guid = 15261848357689602442
define void @WriteNone(i8* %p) #0 {
entry:
  ret void
}

; SSI-LABEL: for function 'Write0'
; SSI: p[]: [0,1)
; BC-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=2 op3=0/>
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "Write0", summaries: {{.*}} params: ((param: 0, offset: [0, 0]))))) ; guid = 5540766144860458461
define void @Write0(i8* %p) #0 {
entry:
  store i8 0, i8* %p
  ret void
}

; SSI-LABEL: for function 'WriteOffset'
; SSI: p[]: [12,16)
; BC-NEXT: <PARAM_ACCESS op0=0 op1=24 op2=32 op3=0/>
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "WriteOffset", summaries: {{.*}} params: ((param: 0, offset: [12, 15]))))) ; guid = 1417835201204712148
define void @WriteOffset(i8* %p) #0 {
entry:
  %0 = bitcast i8* %p to i32*
  %1 = getelementptr i32, i32* %0, i64 3
  store i32 0, i32* %1
  ret void
}

; SSI-LABEL: for function 'WriteNegOffset'
; SSI: p[]: [-56,-48)
; BC-NEXT: <PARAM_ACCESS op0=0 op1=113 op2=97 op3=0/>
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "WriteNegOffset", summaries: {{.*}} params: ((param: 0, offset: [-56, -49]))))) ; guid = 11847411556962310546
define void @WriteNegOffset(i8* %p) #0 {
entry:
  %0 = bitcast i8* %p to i64*
  %1 = getelementptr i64, i64* %0, i64 -7
  store i64 0, i64* %1
  ret void
}

; SSI-LABEL: for function 'WriteAnyOffset'
; SSI: p[]: [-9223372036854775808,9223372036854775807)
; BC-NEXT:  <PARAM_ACCESS op0=0 op1=1 op2=-2 op3=0/>
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "WriteAnyOffset", summaries: {{.*}} params: ((param: 0, offset: [-9223372036854775808, 9223372036854775806]))))) ; guid = 16159595372881907190
define void @WriteAnyOffset(i8* %p, i64 %i) #0 {
entry:
  %0 = bitcast i8* %p to i24*
  %1 = getelementptr i24, i24* %0, i64 %i
  store i24 0, i24* %1
  ret void
}

; SSI-LABEL: for function 'WritePQ'
; SSI: p[]: [0,1)
; SSI: q[]: [0,4)
; BC-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=2 op3=0 op4=1 op5=0 op6=8 op7=0/>
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "WritePQ", summaries: {{.*}} params: ((param: 0, offset: [0, 0]), (param: 1, offset: [0, 3]))))) ; guid = 6187077497926519485
define void @WritePQ(i8* %p, i32* %q) #0 {
entry:
  store i8 5, i8* %p
  store i32 6, i32* %q
  ret void
}

; SSI-LABEL: for function 'WriteTwoPIQ'
; SSI: p[]: [0,1)
; SSI: q[]: [0,4)
; BC-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=2 op3=0 op4=2 op5=0 op6=8 op7=0/>
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "WriteTwoPIQ", summaries: {{.*}} params: ((param: 0, offset: [0, 0]), (param: 2, offset: [0, 3]))))) ; guid = 2949024673554120799
define void @WriteTwoPIQ(i8* %p, i32 %i, i32* %q) #0 {
entry:
  store i8 7, i8* %p
  store i32 %i, i32* %q
  ret void
}

; SSI-LABEL: for function 'Call'
; SSI: p[]: empty-set, @Callee(arg0, [0,1))
; BC-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=0 op3=1 op4=0 op5=[[CALLEE:-?[0-9]+]] op6=0 op7=2/>
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "Call", summaries: {{.*}} calls: ((callee: ^{{.*}})), params: ((param: 0, offset: [0, -1], calls: ((callee: ^{{.*}}, param: 0, offset: [0, 0]))))))) ; guid = 8411925997558855107
define void @Call(i8* %p) #0 {
entry:
  call void @Callee(i8* %p)
  ret void
}

; SSI-LABEL: for function 'CallOffset'
; SSI: p[]: empty-set, @Callee(arg0, [2,3))
; BC-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=0 op3=1 op4=0 op5=[[CALLEE]] op6=4 op7=6/>
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "CallOffset", summaries: {{.*}} calls: ((callee: ^{{.*}})), params: ((param: 0, offset: [0, -1], calls: ((callee: ^{{.*}}, param: 0, offset: [2, 2]))))))) ; guid = 1075564720951610524
define void @CallOffset(i8* %p) #0 {
entry:
  %p1 = getelementptr i8, i8* %p, i64 2
  call void @Callee(i8* %p1)
  ret void
}

; SSI-LABEL: for function 'CallNegOffset'
; SSI: p[]: empty-set, @Callee(arg0, [-715,-714))
; BC-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=0 op3=1 op4=0 op5=[[CALLEE]] op6=1431 op7=1429/>
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "CallNegOffset", summaries: {{.*}} calls: ((callee: ^{{.*}})), params: ((param: 0, offset: [0, -1], calls: ((callee: ^{{.*}}, param: 0, offset: [-715, -715]))))))) ; guid = 16532891468562335146
define void @CallNegOffset(i8* %p) #0 {
entry:
  %p1 = getelementptr i8, i8* %p, i64 -715
  call void @Callee(i8* %p1)
  ret void
}

; BC-NEXT: <PERMODULE
; SSI-LABEL: for function 'CallAnyOffset'
; SSI: p[]: empty-set, @Callee(arg0, full-set)
; DIS-DAG: = gv: (name: "CallAnyOffset", summaries: {{.*}} calls: ((callee: ^{{.*}}))))) ; guid = 4179978066780831873
define void @CallAnyOffset(i8* %p, i64 %i) #0 {
entry:
  %p1 = getelementptr i8, i8* %p, i64 %i
  call void @Callee(i8* %p1)
  ret void
}

; SSI-LABEL: for function 'CallMany'
; SSI: p[]: empty-set, @Callee(arg0, [-715,-714)), @Callee(arg0, [-33,-32)), @Callee(arg0, [124,125))
; BC-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=0 op3=3 op4=0 op5=[[CALLEE]] op6=1431 op7=1429 op8=0 op9=[[CALLEE]] op10=67 op11=65 op12=0 op13=[[CALLEE]] op14=248 op15=250/>
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "CallMany", summaries: {{.*}} calls: ((callee: ^{{.*}})), params: ((param: 0, offset: [0, -1], calls: ((callee: ^{{.*}}, param: 0, offset: [-715, -715]), (callee: ^{{.*}}, param: 0, offset: [-33, -33]), (callee: ^{{.*}}, param: 0, offset: [124, 124]))))))) ; guid = 17150418543861409076
define void @CallMany(i8* %p) #0 {
entry:
  %p0 = getelementptr i8, i8* %p, i64 -715
  call void @Callee(i8* %p0)

  %p1 = getelementptr i8, i8* %p, i64 -33
  call void @Callee(i8* %p1)

  %p2 = getelementptr i8, i8* %p, i64 124
  call void @Callee(i8* %p2)

  ret void
}

; SSI-LABEL: for function 'CallMany2'
; SSI: p[]: empty-set, @Callee(arg0, [-715,-714)), @Callee2(arg1, [-33,-32)), @Callee(arg0, [124,125))
; BC-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=0 op3=3 op4=0 op5=[[CALLEE]] op6=1431 op7=1429 op8=1 op9=[[CALLEE2:-?[0-9]+]] op10=67 op11=65 op12=0 op13=[[CALLEE]] op14=248 op15=250/>
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "CallMany2", summaries: {{.*}} calls: ((callee: ^{{.*}}), (callee: ^{{.*}})), params: ((param: 0, offset: [0, -1], calls: ((callee: ^{{.*}}, param: 0, offset: [-715, -715]), (callee: ^{{.*}}, param: 1, offset: [-33, -33]), (callee: ^{{.*}}, param: 0, offset: [124, 124]))))))) ; guid = 16654048340802466690
define void @CallMany2(i8* %p) #0 {
entry:
  %p0 = getelementptr i8, i8* %p, i64 -715
  call void @Callee(i8* %p0)

  %p1 = getelementptr i8, i8* %p, i64 -33
  call void @Callee2(i32 6, i8* %p1)

  %p2 = getelementptr i8, i8* %p, i64 124
  call void @Callee(i8* %p2)

  ret void
}

; SSI-LABEL: for function 'CallManyUnsafe'
; SSI: p[]: full-set, @Callee(arg0, [-715,-714)), @Callee(arg0, [-33,-32)), @Callee(arg0, [124,125))
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "CallManyUnsafe", summaries: {{.*}} calls: ((callee: ^{{.*}}))))) ; guid = 15696680128757863301
define void @CallManyUnsafe(i8* %p, i64 %i) #0 {
entry:
  %pi = getelementptr i8, i8* %p, i64 %i
  store i8 5, i8* %pi

  %p0 = getelementptr i8, i8* %p, i64 -715
  call void @Callee(i8* %p0)

  %p1 = getelementptr i8, i8* %p, i64 -33
  call void @Callee(i8* %p1)

  %p2 = getelementptr i8, i8* %p, i64 124
  call void @Callee(i8* %p2)

  ret void
}

; SSI-LABEL: for function 'Ret'
; SSI: p[]: full-set
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "Ret", summaries: {{.*}} ; guid = 6707380319572075172
define i8* @Ret(i8* %p) #0 {
entry:
  ret i8* %p
}

; BC-NOT: <PERMODULE
; BC-NOT: <PARAM_ACCESS1


