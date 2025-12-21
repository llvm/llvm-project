; REQUIRES: aarch64-registered-target

; For convenience, to show what is being serialized.
; RUN: opt -S -passes="print<stack-safety-local>" -disable-output < %s 2>&1 | FileCheck %s --check-prefixes=SSI

; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-bcanalyzer -dump %t.bc | FileCheck %s -check-prefixes=BC

; RUN: opt -module-summary %p/Inputs/thinlto-function-summary-paramaccess.ll -o %t2.bc

; RUN: llvm-lto -thinlto -o %t %t.bc %t2.bc

; RUN: llvm-dis -o - %t.thinlto.bc | FileCheck %s --check-prefix=DCO
; Round trip it through llvm-as
; RUN: llvm-dis -o - %t.thinlto.bc | llvm-as -o - | llvm-dis -o - | FileCheck %s --check-prefix=DCO

; RUN: llvm-bcanalyzer -dump %t.thinlto.bc | FileCheck %s --check-prefix=COMBINED

; RUN: llvm-dis -o - %t.bc | FileCheck %s --check-prefix=DIS
; Round trip it through llvm-as
; RUN: llvm-dis -o - %t.bc | llvm-as -o - | llvm-dis -o - | FileCheck %s --check-prefix=DIS

; RUN: opt -thinlto-bc %s -o %t.bc
; RUN: llvm-bcanalyzer -dump %t.bc | FileCheck %s -check-prefixes=BC

; RUN: llvm-dis -o - %t.bc | FileCheck %s --check-prefix=DIS
; Round trip it through llvm-as
; RUN: llvm-dis -o - %t.bc | llvm-as -o - | llvm-dis -o - | FileCheck %s --check-prefix=DIS

; DIS: ^0 = module: (path: "{{.*}}", hash: ({{.*}}))
; DCO: ^0 = module: (path: "{{.*}}", hash: ({{.*}}))
; DCO: ^1 = module: (path: "{{.*}}", hash: ({{.*}}))

; ModuleID = 'thinlto-function-summary-paramaccess.ll'
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux"

attributes #0 = { noinline sanitize_memtag "target-features"="+mte,+neon" }

; BC-LABEL: <GLOBALVAL_SUMMARY_BLOCK
; BC-NEXT: <VERSION
; BC-NEXT: <FLAGS

; DIS-DAG: = gv: (name: "Callee") ; guid = 900789920918863816
; DCO-DAG: = gv: (guid: 900789920918863816, summaries: (function: (module: ^1, flags: ({{[^()]+}}), insts: 1, funcFlags: ({{[^()]+}}), params: ((param: 0, offset: [0, -1]))))){{$}}
 declare void @Callee(i8* %p)

; DIS-DAG: = gv: (name: "Callee2") ; guid = 72710208629861106
; DCO-DAG: = gv: (guid: 72710208629861106, summaries: (function: (module: ^1, flags: ({{[^()]+}}), insts: 1, funcFlags: ({{[^()]+}}), params: ((param: 1, offset: [0, -1]))))){{$}}
 declare void @Callee2(i32 %x, i8* %p)

; BC: <PERMODULE
; DIS-DAG: = gv: (name: "NoParam", summaries: {{.*}} guid = 10287433468618421703
; DCO-DAG: = gv: (guid: 10287433468618421703, summaries: (function: (module: ^0, flags: ({{[^()]+}}), insts: 1, funcFlags: ({{[^()]+}})))){{$}}
define void @NoParam() #0 {
entry:
  ret void
}

; SSI-LABEL: function 'IntParam'
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "IntParam", summaries: {{.*}} guid = 13164714711077064397
; DCO-DAG: = gv: (guid: 13164714711077064397, summaries: (function: (module: ^0, flags: ({{[^()]+}}), insts: 1, funcFlags: ({{[^()]+}})))){{$}}
define void @IntParam(i32 %x) #0 {
entry:
  ret void
}

; SSI-LABEL: for function 'WriteNone'
; SSI: p[]: empty-set
; BC-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=0 op3=0/>
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "WriteNone", summaries: {{.*}} params: ((param: 0, offset: [0, -1]))))) ; guid = 15261848357689602442
; DCO-DAG: = gv: (guid: 15261848357689602442, summaries: (function: (module: ^0, flags: ({{[^()]+}}), insts: 1, funcFlags: ({{[^()]+}}), params: ((param: 0, offset: [0, -1]))))){{$}}
define void @WriteNone(i8* %p) #0 {
entry:
  ret void
}

; SSI-LABEL: for function 'Write0'
; SSI: p[]: [0,1)
; BC-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=2 op3=0/>
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "Write0", summaries: {{.*}} params: ((param: 0, offset: [0, 0]))))) ; guid = 5540766144860458461
; DCO-DAG: = gv: (guid: 5540766144860458461, summaries: (function: (module: ^0, flags: ({{[^()]+}}), insts: 2, funcFlags: ({{[^()]+}}), params: ((param: 0, offset: [0, 0]))))){{$}}
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
; DCO-DAG: = gv: (guid: 1417835201204712148, summaries: (function: (module: ^0, flags: ({{[^()]+}}), insts: 4, funcFlags: ({{[^()]+}}), params: ((param: 0, offset: [12, 15]))))){{$}}
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
; DCO-DAG: = gv: (guid: 11847411556962310546, summaries: (function: (module: ^0, flags: ({{[^()]+}}), insts: 4, funcFlags: ({{[^()]+}}), params: ((param: 0, offset: [-56, -49]))))){{$}}
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
; DCO-DAG: = gv: (guid: 16159595372881907190, summaries: (function: (module: ^0, flags: ({{[^()]+}}), insts: 4, funcFlags: ({{[^()]+}}), params: ((param: 0, offset: [-9223372036854775808, 9223372036854775806]))))){{$}}
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
; DCO-DAG: = gv: (guid: 6187077497926519485, summaries: (function: (module: ^0, flags: ({{[^()]+}}), insts: 3, funcFlags: ({{[^()]+}}), params: ((param: 0, offset: [0, 0]), (param: 1, offset: [0, 3]))))){{$}}
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
; DCO-DAG: = gv: (guid: 2949024673554120799, summaries: (function: (module: ^0, flags: ({{[^()]+}}), insts: 3, funcFlags: ({{[^()]+}}), params: ((param: 0, offset: [0, 0]), (param: 2, offset: [0, 3]))))){{$}}
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
; DCO-DAG: = gv: (guid: 8411925997558855107, summaries: (function: (module: ^0, flags: ({{[^()]+}}), insts: 2, funcFlags: ({{[^()]+}}), calls: ((callee: ^[[CALLEE:.]])), params: ((param: 0, offset: [0, -1], calls: ((callee: ^[[CALLEE]], param: 0, offset: [0, 0]))))))){{$}}
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
; DCO-DAG: = gv: (guid: 1075564720951610524, summaries: (function: (module: ^0, flags: ({{[^()]+}}), insts: 3, funcFlags: ({{[^()]+}}), calls: ((callee: ^[[CALLEE:.]])), params: ((param: 0, offset: [0, -1], calls: ((callee: ^[[CALLEE]], param: 0, offset: [2, 2]))))))){{$}}
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
; DCO-DAG: = gv: (guid: 16532891468562335146, summaries: (function: (module: ^0, flags: ({{[^()]+}}), insts: 3, funcFlags: ({{[^()]+}}), calls: ((callee: ^[[CALLEE:.]])), params: ((param: 0, offset: [0, -1], calls: ((callee: ^[[CALLEE]], param: 0, offset: [-715, -715]))))))){{$}}
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
; DCO-DAG: = gv: (guid: 4179978066780831873, summaries: (function: (module: ^0, flags: ({{[^()]+}}), insts: 3, funcFlags: ({{[^()]+}}), calls: ((callee: ^[[CALLEE:.]]))))){{$}}
define void @CallAnyOffset(i8* %p, i64 %i) #0 {
entry:
  %p1 = getelementptr i8, i8* %p, i64 %i
  call void @Callee(i8* %p1)
  ret void
}

; SSI-LABEL: for function 'CallMany'
; SSI: p[]: empty-set, @Callee(arg0, [-715,125))
; BC-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=0 op3=1 op4=0 op5=[[CALLEE]] op6=1431 op7=250/>
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "CallMany", summaries: {{.*}} calls: ((callee: ^{{.*}})), params: ((param: 0, offset: [0, -1], calls: ((callee: ^{{.*}}, param: 0, offset: [-715, 124]))))))) ; guid = 17150418543861409076
; DCO-DAG: = gv: (guid: 17150418543861409076, summaries: (function: (module: ^0, flags: ({{[^()]+}}), insts: 7, funcFlags: ({{[^()]+}}), calls: ((callee: ^[[CALLEE:.]])), params: ((param: 0, offset: [0, -1], calls: ((callee: ^[[CALLEE]], param: 0, offset: [-715, 124]))))))){{$}}
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
; SSI: p[]: empty-set, @Callee(arg0, [-715,125)), @Callee2(arg1, [-33,-32))
; BC-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=0 op3=2 op4=0 op5=[[CALLEE]] op6=1431 op7=250 op8=1 op9=[[CALLEE2:-?[0-9]+]] op10=67 op11=65/>
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "CallMany2", summaries: {{.*}} calls: ((callee: ^{{.*}}), (callee: ^{{.*}})), params: ((param: 0, offset: [0, -1], calls: ((callee: ^{{.*}}, param: 0, offset: [-715, 124]), (callee: ^{{.*}}, param: 1, offset: [-33, -33]))))))) ; guid = 16654048340802466690
; DCO-DAG: = gv: (guid: 16654048340802466690, summaries: (function: (module: ^0, flags: ({{[^()]+}}), insts: 7, funcFlags: ({{[^()]+}}), calls: ((callee: ^{{[0-9]+}}), (callee: ^{{[0-9]+}})), params: ((param: 0, offset: [0, -1], calls: ((callee: ^{{[0-9]+}}, param: 0, offset: [-715, 124]), (callee: ^{{[0-9]+}}, param: 1, offset: [-33, -33]))))))){{$}}
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
; SSI: p[]: full-set, @Callee(arg0, [-715,125))
; BC-NEXT: <PERMODULE
; DIS-DAG: = gv: (name: "CallManyUnsafe", summaries: {{.*}} calls: ((callee: ^{{.*}}))))) ; guid = 15696680128757863301
; DCO-DAG: = gv: (guid: 15696680128757863301, summaries: (function: (module: ^0, flags: ({{[^()]+}}), insts: 9, funcFlags: ({{[^()]+}}), calls: ((callee: ^[[CALLEE:.]]))))){{$}}
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
; DCO-DAG: = gv: (guid: 6707380319572075172, summaries: (function: (module: ^0, flags: ({{[^()]+}}), insts: 1, funcFlags: ({{[^()]+}})))){{$}}
define i8* @Ret(i8* %p) #0 {
entry:
  ret i8* %p
}

; BC-NOT: <PERMODULE
; BC-NOT: <PARAM_ACCESS1


; COMBINED: <FLAGS op0=0/>
; COMBINED-NEXT: <VALUE_GUID {{.*}} op0=[[CALLEE1:1]] op1=16929164 op2=2901240562/>
; COMBINED-NEXT: <VALUE_GUID {{.*}} op0=[[CALLEE2:2]] op1=209731497 op2=362741704/>
; COMBINED-NEXT: <VALUE_GUID {{.*}} op0=3 op1=250424426 op2=1162038428/>
; COMBINED-NEXT: <VALUE_GUID {{.*}} op0=4 op1=330115482 op2=2111435476/>
; COMBINED-NEXT: <VALUE_GUID {{.*}} op0=5 op1=686623312 op2=3842916447/>
; COMBINED-NEXT: <VALUE_GUID {{.*}} op0=6 op1=973226983 op2=3211083905/>
; COMBINED-NEXT: <VALUE_GUID {{.*}} op0=7 op1=1290060147 op2=3622505949/>
; COMBINED-NEXT: <VALUE_GUID {{.*}} op0=8 op1=1440541236 op2=767101629/>
; COMBINED-NEXT: <VALUE_GUID {{.*}} op0=9 op1=1561683677 op2=160047780/>
; COMBINED-NEXT: <VALUE_GUID {{.*}} op0=10 op1=1958554144 op2=1633580483/>
; COMBINED-NEXT: <VALUE_GUID {{.*}} op0=11 op1=2395229756 op2=192361927/>
; COMBINED-NEXT: <VALUE_GUID {{.*}} op0=12 op1=2758440458 op2=1889048978/>
; COMBINED-NEXT: <VALUE_GUID {{.*}} op0=13 op1=3065148999 op2=3004927693/>
; COMBINED-NEXT: <VALUE_GUID {{.*}} op0=14 op1=3553425976 op2=2012721546/>
; COMBINED-NEXT: <VALUE_GUID {{.*}} op0=15 op1=3654668137 op2=2609615749/>
; COMBINED-NEXT: <VALUE_GUID {{.*}} op0=16 op1=3762448991 op2=3668708854/>
; COMBINED-NEXT: <VALUE_GUID {{.*}} op0=17 op1=3849363762 op2=364807594/>
; COMBINED-NEXT: <VALUE_GUID {{.*}} op0=18 op1=3877572794 op2=2713121666/>
; COMBINED-NEXT: <VALUE_GUID {{.*}} op0=19 op1=3993142988 op2=2149688628/>
; COMBINED-NEXT: <PARAM_ACCESS op0=1 op1=0 op2=0 op3=0/>
; COMBINED-NEXT: <COMBINED_PROFILE {{.*}} op0=1
; COMBINED-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=0 op3=0/>
; COMBINED-NEXT: <COMBINED_PROFILE {{.*}} op0=2
; COMBINED-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=0 op3=1 op4=0 op5=[[CALLEE2]] op6=4 op7=6/>
; COMBINED-NEXT: <COMBINED_PROFILE {{.*}} op0=3
; COMBINED-NEXT: <PARAM_ACCESS op0=0 op1=24 op2=32 op3=0/>
; COMBINED-NEXT: <COMBINED_PROFILE {{.*}} op0=4
; COMBINED-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=2 op3=0 op4=2 op5=0 op6=8 op7=0/>
; COMBINED-NEXT: <COMBINED_PROFILE {{.*}} op0=5
; COMBINED-NEXT: <COMBINED_PROFILE {{.*}} op0=6
; COMBINED-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=2 op3=0/>
; COMBINED-NEXT: <COMBINED_PROFILE {{.*}} op0=7
; COMBINED-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=2 op3=0 op4=1 op5=0 op6=8 op7=0/>
; COMBINED-NEXT: <COMBINED_PROFILE {{.*}} op0=8
; COMBINED-NEXT: <COMBINED_PROFILE {{.*}} op0=9
; COMBINED-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=0 op3=1 op4=0 op5=[[CALLEE2]] op6=0 op7=2/>
; COMBINED-NEXT: <COMBINED_PROFILE {{.*}} op0=10
; COMBINED-NEXT: <COMBINED_PROFILE {{.*}} op0=11
; COMBINED-NEXT: <PARAM_ACCESS op0=0 op1=113 op2=97 op3=0/>
; COMBINED-NEXT: <COMBINED_PROFILE {{.*}} op0=12
; COMBINED-NEXT: <COMBINED_PROFILE {{.*}} op0=13
; COMBINED-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=0 op3=0/>
; COMBINED-NEXT: <COMBINED_PROFILE {{.*}} op0=14
; COMBINED-NEXT: <COMBINED_PROFILE {{.*}} op0=15
; COMBINED-NEXT: <PARAM_ACCESS op0=0 op1=1 op2=-2 op3=0/>
; COMBINED-NEXT: <COMBINED_PROFILE {{.*}} op0=16
; COMBINED-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=0 op3=1 op4=0 op5=[[CALLEE2]] op6=1431 op7=1429/>
; COMBINED-NEXT: <COMBINED_PROFILE {{.*}} op0=17
; COMBINED-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=0 op3=2 op4=0 op5=[[CALLEE2]] op6=1431 op7=250 op8=1 op9=[[CALLEE1]] op10=67 op11=65/>
; COMBINED-NEXT: <COMBINED_PROFILE {{.*}} op0=18
; COMBINED-NEXT: <PARAM_ACCESS op0=0 op1=0 op2=0 op3=1 op4=0 op5=[[CALLEE2]] op6=1431 op7=250/>
; COMBINED-NEXT: <COMBINED_PROFILE {{.*}} op0=19
