
; RUN: llvm-profdata merge %S/Inputs/misexpect-branch.proftext -o %t.profdata


; RUN: opt < %s -lower-expect -pgo-instr-use -pgo-test-profile-file=%t.profdata -S -pgo-warn-misexpect 2>&1 | FileCheck %s --check-prefix=WARNING
; RUN: opt < %s -lower-expect -pgo-instr-use -pgo-test-profile-file=%t.profdata -S -pass-remarks=misexpect 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: opt < %s -lower-expect -pgo-instr-use -pgo-test-profile-file=%t.profdata -S -pgo-warn-misexpect -pass-remarks=misexpect 2>&1 | FileCheck %s --check-prefix=BOTH
; RUN: opt < %s -lower-expect -pgo-instr-use -pgo-test-profile-file=%t.profdata -S 2>&1 | FileCheck %s --check-prefix=DISABLED

; New PM
; RUN: opt < %s -passes="function(lower-expect),pgo-instr-use" -pgo-test-profile-file=%t.profdata -pgo-warn-misexpect -S 2>&1 | FileCheck %s --check-prefix=WARNING
; RUN: opt < %s -passes="function(lower-expect),pgo-instr-use" -pgo-test-profile-file=%t.profdata -pass-remarks=misexpect -S 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: opt < %s -passes="function(lower-expect),pgo-instr-use" -pgo-test-profile-file=%t.profdata -pgo-warn-misexpect -pass-remarks=misexpect -S 2>&1 | FileCheck %s --check-prefix=BOTH
; RUN: opt < %s -passes="function(lower-expect),pgo-instr-use" -pgo-test-profile-file=%t.profdata -S 2>&1 | FileCheck %s --check-prefix=DISABLED



; WARNING-DAG: warning: <unknown>:0:0: 19.98%
; WARNING-NOT: remark: <unknown>:0:0: Potential performance regression from use of the llvm.expect intrinsic: Annotation was correct on 19.98% (399668 / 2000000) of profiled executions.

; REMARK-NOT: warning: <unknown>:0:0: 19.98%
; REMARK-DAG: remark: <unknown>:0:0: Potential performance regression from use of the llvm.expect intrinsic: Annotation was correct on 19.98% (399668 / 2000000) of profiled executions.

; BOTH-DAG: warning: <unknown>:0:0: 19.98%
; BOTH-DAG: remark: <unknown>:0:0: Potential performance regression from use of the llvm.expect intrinsic: Annotation was correct on 19.98% (399668 / 2000000) of profiled executions.

; DISABLED-NOT: warning: <unknown>:0:0: 19.98%
; DISABLED-NOT: remark: <unknown>:0:0: Potential performance regression from use of the llvm.expect intrinsic: Annotation was correct on 19.98% (399668 / 2000000) of profiled executions.

; CHECK-DAG: !{!"misexpect", i64 1, i64 2000, i64 1}



; ModuleID = 'misexpect-branch.c'
source_filename = "misexpect-branch.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@inner_loop = constant i32 100, align 4
@outer_loop = constant i32 2000, align 4

; Function Attrs: nounwind
define i32 @bar() #0 {
entry:
  %rando = alloca i32, align 4
  %x = alloca i32, align 4
  %0 = bitcast i32* %rando to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #4
  %call = call i32 (...) @buzz()
  store i32 %call, i32* %rando, align 4, !tbaa !3
  %1 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %1) #4
  store i32 0, i32* %x, align 4, !tbaa !3
  %2 = load i32, i32* %rando, align 4, !tbaa !3
  %rem = srem i32 %2, 200000
  %cmp = icmp eq i32 %rem, 0
  %lnot = xor i1 %cmp, true
  %lnot1 = xor i1 %lnot, true
  %lnot.ext = zext i1 %lnot1 to i32
  %conv = sext i32 %lnot.ext to i64
  %expval = call i64 @llvm.expect.i64(i64 %conv, i64 1)
  %tobool = icmp ne i64 %expval, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %3 = load i32, i32* %rando, align 4, !tbaa !3
  %call2 = call i32 @baz(i32 %3)
  store i32 %call2, i32* %x, align 4, !tbaa !3
  br label %if.end

if.else:                                          ; preds = %entry
  %call3 = call i32 @foo(i32 50)
  store i32 %call3, i32* %x, align 4, !tbaa !3
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %4 = load i32, i32* %x, align 4, !tbaa !3
  %5 = bitcast i32* %x to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %5) #4
  %6 = bitcast i32* %rando to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %6) #4
  ret i32 %4
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

declare i32 @buzz(...) #2

; Function Attrs: nounwind readnone willreturn
declare i64 @llvm.expect.i64(i64, i64) #3

declare i32 @baz(i32) #2

declare i32 @foo(i32) #2

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone willreturn }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{!"clang version 10.0.0 (trunk c20270bfffc9d6965219de339d66c61e9fe7d82d)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
