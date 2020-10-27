; RUN: opt < %s -instrprof -S -o - -do-counter-promotion=1  -skip-ret-exit-block=1 | FileCheck %s --check-prefixes=CHECK,SKIP
; RUN: opt < %s -instrprof -S -o - -do-counter-promotion=1  -skip-ret-exit-block=0 | FileCheck %s --check-prefixes=CHECK,NOTSKIP

$__llvm_profile_raw_version = comdat any

@bar = dso_local local_unnamed_addr global i32 0, align 4
@__llvm_profile_raw_version = constant i64 72057594037927941, comdat
@__profn_foo = private constant [3 x i8] c"foo"

define dso_local void @foo(i32 %n)  {
entry:
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 29212902728, i32 2, i32 1)
  br label %for.cond

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %if.end ]
  %cmp = icmp slt i32 %i.0, %n
  %0 = load i32, i32* @bar, align 4
  %tobool.not = icmp eq i32 %0, 0
  %or.cond = and i1 %cmp, %tobool.not
  br i1 %or.cond, label %if.end, label %cleanup

if.end:
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 29212902728, i32 2, i32 0)
  call void (...) @bar2()
  %inc = add nuw nsw i32 %i.0, 1
  br label %for.cond

cleanup:
; CHECK: cleanup:
; SKIP-NOT:  %pgocount.promoted
; NOTSKIP:  %pgocount.promoted
  ret void
}

declare dso_local void @bar2(...)

declare void @llvm.instrprof.increment(i8*, i64, i32, i32)
