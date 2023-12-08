; Test misexpect handles switch statements when debug information is stripped

; RUN: llvm-profdata merge %S/Inputs/misexpect-switch.proftext -o %t.profdata

; RUN: opt < %s -passes="function(lower-expect),pgo-instr-use" -pgo-test-profile-file=%t.profdata -pgo-warn-misexpect -S 2>&1 | FileCheck %s --check-prefix=WARNING
; RUN: opt < %s -passes="function(lower-expect),pgo-instr-use" -pgo-test-profile-file=%t.profdata -pass-remarks=misexpect -S 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: opt < %s -passes="function(lower-expect),pgo-instr-use" -pgo-test-profile-file=%t.profdata -pgo-warn-misexpect -pass-remarks=misexpect -S 2>&1 | FileCheck %s --check-prefix=BOTH
; RUN: opt < %s -passes="function(lower-expect),pgo-instr-use" -pgo-test-profile-file=%t.profdata -S 2>&1 | FileCheck %s --check-prefix=DISABLED

; WARNING-DAG: warning: <unknown>:0:0: 0.00%
; WARNING-NOT: remark: <unknown>:0:0: Potential performance regression from use of the llvm.expect intrinsic: Annotation was correct on 0.00% (0 / 27943) of profiled executions.

; REMARK-NOT: warning: <unknown>:0:0: 0.00%
; REMARK-DAG: remark: <unknown>:0:0: Potential performance regression from use of the llvm.expect intrinsic: Annotation was correct on 0.00% (0 / 27943) of profiled executions.

; BOTH-DAG: warning: <unknown>:0:0: 0.00%
; BOTH-DAG: remark: <unknown>:0:0: Potential performance regression from use of the llvm.expect intrinsic: Annotation was correct on 0.00% (0 / 27943) of profiled executions.

; DISABLED-NOT: warning: <unknown>:0:0: 0.00%
; DISABLED-NOT: remark: <unknown>:0:0: Potential performance regression from use of the llvm.expect intrinsic: Annotation was correct on 0.00% (0 / 27943) of profiled executions.

; DISABLED-NOT: warning: <unknown>:0:0: 0.00%
; DISABLED-NOT: remark: <unknown>:0:0: Potential performance regression from use of the llvm.expect intrinsic: Annotation was correct on 0.00% (0 / 27943) of profiled executions.

; CORRECT-NOT: warning: {{.*}}
; CORRECT-NOT: remark: {{.*}}


; ModuleID = 'misexpect-switch.c'
source_filename = "misexpect-switch.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@inner_loop = dso_local constant i32 1000, align 4
@outer_loop = dso_local constant i32 20, align 4
@arry_size = dso_local constant i32 25, align 4
@arry = dso_local global [25 x i32] zeroinitializer, align 16

; Function Attrs: nounwind uwtable
define dso_local void @init_arry() #0 {
entry:
  %i = alloca i32, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr %i) #6
  store i32 0, ptr %i, align 4, !tbaa !4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4, !tbaa !4
  %cmp = icmp slt i32 %0, 25
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %call = call i32 @rand() #6
  %rem = srem i32 %call, 10
  %1 = load i32, ptr %i, align 4, !tbaa !4
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [25 x i32], ptr @arry, i64 0, i64 %idxprom
  store i32 %rem, ptr %arrayidx, align 4, !tbaa !4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %2 = load i32, ptr %i, align 4, !tbaa !4
  %inc = add nsw i32 %2, 1
  store i32 %inc, ptr %i, align 4, !tbaa !4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  call void @llvm.lifetime.end.p0(i64 4, ptr %i) #6
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: nounwind
declare dso_local i32 @rand() #3

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %val = alloca i32, align 4
  %j = alloca i32, align 4
  %condition = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  call void @init_arry()
  call void @llvm.lifetime.start.p0(i64 4, ptr %val) #6
  store i32 0, ptr %val, align 4, !tbaa !4
  call void @llvm.lifetime.start.p0(i64 4, ptr %j) #6
  store i32 0, ptr %j, align 4, !tbaa !4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %j, align 4, !tbaa !4
  %cmp = icmp slt i32 %0, 20000
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0(i64 4, ptr %condition) #6
  %call = call i32 @rand() #6
  %rem = srem i32 %call, 5
  store i32 %rem, ptr %condition, align 4, !tbaa !4
  %1 = load i32, ptr %condition, align 4, !tbaa !4
  %conv = zext i32 %1 to i64
  %expval = call i64 @llvm.expect.i64(i64 %conv, i64 6)
  switch i64 %expval, label %sw.default [
    i64 0, label %sw.bb
    i64 1, label %sw.bb2
    i64 2, label %sw.bb2
    i64 3, label %sw.bb2
    i64 4, label %sw.bb3
  ]

sw.bb:                                            ; preds = %for.body
  %call1 = call i32 @sum(ptr @arry, i32 25)
  %2 = load i32, ptr %val, align 4, !tbaa !4
  %add = add nsw i32 %2, %call1
  store i32 %add, ptr %val, align 4, !tbaa !4
  br label %sw.epilog

sw.bb2:                                           ; preds = %for.body, %for.body, %for.body
  br label %sw.epilog

sw.bb3:                                           ; preds = %for.body
  %call4 = call i32 @random_sample(ptr @arry, i32 25)
  %3 = load i32, ptr %val, align 4, !tbaa !4
  %add5 = add nsw i32 %3, %call4
  store i32 %add5, ptr %val, align 4, !tbaa !4
  br label %sw.epilog

sw.default:                                       ; preds = %for.body
  unreachable

sw.epilog:                                        ; preds = %sw.bb3, %sw.bb2, %sw.bb
  call void @llvm.lifetime.end.p0(i64 4, ptr %condition) #6
  br label %for.inc

for.inc:                                          ; preds = %sw.epilog
  %4 = load i32, ptr %j, align 4, !tbaa !4
  %inc = add nsw i32 %4, 1
  store i32 %inc, ptr %j, align 4, !tbaa !4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  call void @llvm.lifetime.end.p0(i64 4, ptr %j) #6
  call void @llvm.lifetime.end.p0(i64 4, ptr %val) #6
  ret i32 0
}

; Function Attrs: nounwind readnone willreturn
declare i64 @llvm.expect.i64(i64, i64) #4

declare dso_local i32 @sum(ptr, i32) #5

declare dso_local i32 @random_sample(ptr, i32) #5

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { nounwind readnone speculatable willreturn }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind readnone willreturn }
attributes #5 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{!"clang version 10.0.0 (60b79b85b1763d3d25630261e5cd1adb7f0835bc)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
