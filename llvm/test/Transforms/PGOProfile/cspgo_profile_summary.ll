; Test the profile summary for context sensitive PGO (CSPGO)

; RUN: llvm-profdata merge %S/Inputs/cspgo.proftext -o %t.profdata
; RUN: opt < %s -passes='default<O2>' -disable-preinline -pgo-instrument-entry=false -pgo-kind=pgo-instr-use-pipeline -profile-file=%t.profdata -S | FileCheck %s --check-prefix=PGOSUMMARY
; RUN: opt < %s -O2 -disable-preinline -pgo-instrument-entry=false -pgo-kind=pgo-instr-use-pipeline -profile-file=%t.profdata -S | FileCheck %s --check-prefix=PGOSUMMARY
; RUN: opt < %s -O2 -disable-preinline -pgo-instrument-entry=false -pgo-kind=pgo-instr-use-pipeline -profile-file=%t.profdata -S -cspgo-kind=cspgo-instr-use-pipeline| FileCheck %s --check-prefix=CSPGOSUMMARY

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@odd = common dso_local global i32 0, align 4
@even = common dso_local global i32 0, align 4
@not_six = common dso_local global i32 0, align 4

define dso_local i32 @goo(i32 %n) {
entry:
  %i = alloca i32, align 4
  %i.0..sroa_cast = bitcast i32* %i to i8*
  store volatile i32 %n, i32* %i, align 4
  %i.0. = load volatile i32, i32* %i, align 4
  ret i32 %i.0.
}

define dso_local void @bar(i32 %n) {
entry:
  %call = call fastcc i32 @cond(i32 %n)
  %tobool = icmp eq i32 %call, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:
  %0 = load i32, i32* @odd, align 4
  %inc = add i32 %0, 1
  store i32 %inc, i32* @odd, align 4
  br label %if.end

if.else:
  %1 = load i32, i32* @even, align 4
  %inc1 = add i32 %1, 1
  store i32 %inc1, i32* @even, align 4
  br label %if.end

if.end:
  br label %for.cond

for.cond:
  %i.0 = phi i32 [ 0, %if.end ], [ %inc6, %for.inc ]
  %cmp = icmp ult i32 %i.0, 4
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %mul = mul nsw i32 %i.0, %n
  %rem = srem i32 %mul, 6
  %tobool2 = icmp eq i32 %rem, 0
  br i1 %tobool2, label %for.inc, label %if.then3

if.then3:
  %2 = load i32, i32* @not_six, align 4
  %inc4 = add i32 %2, 1
  store i32 %inc4, i32* @not_six, align 4
  br label %for.inc

for.inc:
  %inc6 = add nuw nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret void
}
; PGOSUMMARY-LABEL: @bar
; PGOSUMMARY: %odd.sink{{[0-9]*}} = select i1 %tobool{{[0-9]*}}, i32* @even, i32* @odd
; PGOSUMMARY-SAME: !prof ![[BW_PGO_BAR:[0-9]+]]
; CSPGOSUMMARY-LABEL: @bar
; CSPGOSUMMARY: %odd.sink{{[0-9]*}} = select i1 %tobool{{[0-9]*}}, i32* @even, i32* @odd
; CSPGOSUMMARY-SAME: !prof ![[BW_CSPGO_BAR:[0-9]+]]

define internal fastcc i32 @cond(i32 %i) {
entry:
  %rem = srem i32 %i, 2
  ret i32 %rem
}

define dso_local void @foo() {
entry:
  br label %for.cond

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %add4, %for.body ]
  %cmp = icmp slt i32 %i.0, 200000
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %call = call i32 @goo(i32 %i.0)
  call void @bar(i32 %call)
  %add = add nsw i32 %call, 1
  call void @bar(i32 %add)
  %call1 = call i32 @bar_m(i32 %call) #4
  %call3 = call i32 @bar_m2(i32 %add) #4
  call fastcc void @barbar()
  %add4 = add nsw i32 %call, 2
  br label %for.cond

for.end:
  ret void
}
; CSPGOSUMMARY-LABEL: @foo
; CSPGOSUMMARY: %even.sink{{[0-9]*}} = select i1 %tobool.i{{[0-9]*}}, i32* @even, i32* @odd
; CSPGOSUMMARY-SAME: !prof ![[BW1_CSPGO_FOO:[0-9]+]]
; CSPGOSUMMARY: %even.sink{{[0-9]*}} = select i1 %tobool.i{{[0-9]*}}, i32* @even, i32* @odd
; CSPGOSUMMARY-SAME: !prof ![[BW2_CSPGO_FOO:[0-9]+]]

declare dso_local i32 @bar_m(i32)
declare dso_local i32 @bar_m2(i32)

define internal fastcc void @barbar() {
entry:
  %0 = load i32, i32* @odd, align 4
  %inc = add i32 %0, 1
  store i32 %inc, i32* @odd, align 4
  ret void
}

define dso_local i32 @main() {
entry:
  call void @foo()
  ret i32 0
}

; PGOSUMMARY: {{![0-9]+}} = !{i32 1, !"ProfileSummary", !{{[0-9]+}}}
; PGOSUMMARY: {{![0-9]+}} = !{!"ProfileFormat", !"InstrProf"}
; PGOSUMMARY: {{![0-9]+}} = !{!"TotalCount", i64 2100001}
; PGOSUMMARY: {{![0-9]+}} = !{!"MaxCount", i64 800000}
; PGOSUMMARY: {{![0-9]+}} = !{!"MaxInternalCount", i64 399999}
; PGOSUMMARY: {{![0-9]+}} = !{!"MaxFunctionCount", i64 800000}
; PGOSUMMARY: {{![0-9]+}} = !{!"NumCounts", i64 14}
; PGOSUMMARY: {{![0-9]+}} = !{!"NumFunctions", i64 8}
; PGOSUMMARY-DAG: ![[BW_PGO_BAR]] = !{!"branch_weights", i32 100000, i32 100000}

; CSPGOSUMMARY: {{![0-9]+}} = !{i32 1, !"ProfileSummary", !1}
; CSPGOSUMMARY: {{![0-9]+}} = !{!"ProfileFormat", !"InstrProf"}
; CSPGOSUMMARY: {{![0-9]+}} = !{!"TotalCount", i64 2100001}
; CSPGOSUMMARY: {{![0-9]+}} = !{!"MaxCount", i64 800000}
; CSPGOSUMMARY: {{![0-9]+}} = !{!"MaxInternalCount", i64 399999}
; CSPGOSUMMARY: {{![0-9]+}} = !{!"MaxFunctionCount", i64 800000}
; CSPGOSUMMARY: {{![0-9]+}} = !{!"NumCounts", i64 14}
; CSPGOSUMMARY: {{![0-9]+}} = !{!"NumFunctions", i64 8}
; CSPGOSUMMARY: {{![0-9]+}} = !{!"DetailedSummary", !{{[0-9]+}}}
; CSPGOSUMMARY: {{![0-9]+}} = !{i32 1, !"CSProfileSummary", !{{[0-9]+}}}
; CSPGOSUMMARY: {{![0-9]+}} = !{!"ProfileFormat", !"CSInstrProf"}
; CSPGOSUMMARY: {{![0-9]+}} = !{!"TotalCount", i64 1299950}
; CSPGOSUMMARY: {{![0-9]+}} = !{!"MaxCount", i64 200000}
; CSPGOSUMMARY: {{![0-9]+}} = !{!"MaxInternalCount", i64 100000}
; CSPGOSUMMARY: {{![0-9]+}} = !{!"MaxFunctionCount", i64 200000}
; CSPGOSUMMARY: {{![0-9]+}} = !{!"NumCounts", i64 23}
; CSPGOSUMMARY-DAG: ![[BW_CSPGO_BAR]] = !{!"branch_weights", i32 100000, i32 100000}
; CSPGOSUMMARY-DAG: ![[BW1_CSPGO_FOO]] = !{!"branch_weights", i32 100000, i32 0}
; CSPGOSUMMARY-DAG: ![[BW2_CSPGO_FOO]] = !{!"branch_weights", i32 0, i32 100000}
