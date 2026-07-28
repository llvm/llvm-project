; Test misexpect checks do not issue diagnostics when profiling weights and
; branch weights added by llvm.expect agree

; RUN: llvm-profdata merge %S/Inputs/misexpect-branch-correct.proftext -o %t.profdata

; RUN: opt < %s -passes="function(lower-expect),pgo-instr-use" -pgo-test-profile-file=%t.profdata -pgo-warn-misexpect -pass-remarks=misexpect -S  2>&1 | FileCheck %s

; CHECK-NOT: warning: {{.*}}
; CHECK-NOT: remark: {{.*}}
; CHECK: !{!"branch_weights", i32 0, i32 200000}

; ModuleID = 'misexpect-branch-correct.c'
source_filename = "misexpect-branch-correct.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@inner_loop = constant i32 100, align 4
@outer_loop = constant i32 2000, align 4

; Function Attrs: nounwind
define i32 @bar() {
entry:
  %rando = alloca i32, align 4
  %x = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr %rando)
  %call = call i32 (...) @buzz()
  store i32 %call, ptr %rando, align 4, !tbaa !3
  call void @llvm.lifetime.start.p0(ptr %x)
  store i32 0, ptr %x, align 4, !tbaa !3
  %0 = load i32, ptr %rando, align 4, !tbaa !3
  %rem = srem i32 %0, 200000
  %cmp = icmp eq i32 %rem, 0
  %lnot = xor i1 %cmp, true
  %lnot1 = xor i1 %lnot, true
  %lnot.ext = zext i1 %lnot1 to i32
  %conv = sext i32 %lnot.ext to i64
  %expval = call i64 @llvm.expect.i64(i64 %conv, i64 0)
  %tobool = icmp ne i64 %expval, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %1 = load i32, ptr %rando, align 4, !tbaa !3
  %call2 = call i32 @baz(i32 %1)
  store i32 %call2, ptr %x, align 4, !tbaa !3
  br label %if.end

if.else:                                          ; preds = %entry
  %call3 = call i32 @foo(i32 50)
  store i32 %call3, ptr %x, align 4, !tbaa !3
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %2 = load i32, ptr %x, align 4, !tbaa !3
  call void @llvm.lifetime.end.p0(ptr %x)
  call void @llvm.lifetime.end.p0(ptr %rando)
  ret i32 %2
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(ptr nocapture)

declare i32 @buzz(...)

; Function Attrs: nounwind readnone willreturn
declare i64 @llvm.expect.i64(i64, i64)

declare i32 @baz(i32)

declare i32 @foo(i32)

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(ptr nocapture)

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{!"clang version 10.0.0 (c20270bfffc9d6965219de339d66c61e9fe7d82d)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
