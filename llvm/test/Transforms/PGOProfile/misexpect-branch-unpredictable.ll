; Test misexpect doesn't issue diagnostics when a branch is marked unpredictable

; RUN: llvm-profdata merge %S/Inputs/misexpect-branch-correct.proftext -o %t.profdata

; RUN: opt < %s -passes="function(lower-expect),pgo-instr-use" -pgo-test-profile-file=%t.profdata -pgo-warn-misexpect -pass-remarks=misexpect -S  2>&1 | FileCheck %s

; CHECK-NOT: warning: {{.*}}
; CHECK-NOT: remark: {{.*}}


; ModuleID = 'misexpect-branch-unpredictable.c'
source_filename = "clang/test/Profile/misexpect-branch-unpredictable.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@inner_loop = constant i32 100, align 4
@outer_loop = constant i32 2000, align 4

; Function Attrs: nounwind
define i32 @bar() #0 {
entry:
  %rando = alloca i32, align 4
  %x = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr %rando) #3
  %call = call i32 (...) @buzz()
  store i32 %call, ptr %rando, align 4, !tbaa !2
  call void @llvm.lifetime.start.p0(ptr %x) #3
  store i32 0, ptr %x, align 4, !tbaa !2
  %0 = load i32, ptr %rando, align 4, !tbaa !2
  %rem = srem i32 %0, 200000
  %cmp = icmp eq i32 %rem, 0
  %lnot = xor i1 %cmp, true
  %lnot1 = xor i1 %lnot, true
  %lnot.ext = zext i1 %lnot1 to i32
  %conv = sext i32 %lnot.ext to i64
  %tobool = icmp ne i64 %conv, 0
  br i1 %tobool, label %if.then, label %if.else, !unpredictable !6

if.then:                                          ; preds = %entry
  %1 = load i32, ptr %rando, align 4, !tbaa !2
  %call2 = call i32 @baz(i32 %1)
  store i32 %call2, ptr %x, align 4, !tbaa !2
  br label %if.end

if.else:                                          ; preds = %entry
  %call3 = call i32 @foo(i32 50)
  store i32 %call3, ptr %x, align 4, !tbaa !2
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %2 = load i32, ptr %x, align 4, !tbaa !2
  call void @llvm.lifetime.end.p0(ptr %x) #3
  call void @llvm.lifetime.end.p0(ptr %rando) #3
  ret i32 %2
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(ptr nocapture) #1

declare i32 @buzz(...) #2

declare i32 @baz(i32) #2

declare i32 @foo(i32) #2

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(ptr nocapture) #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"Fuchsia clang version 10.0.0 (153b453014c94291c8c6cf6320b2f46df40f26f3) (based on LLVM 10.0.0svn)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{}
