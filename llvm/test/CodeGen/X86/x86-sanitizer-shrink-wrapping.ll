; RUN: llc -o - < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "x86_64-apple-macosx"

; Even if the target supports shrink-wrapping, the prologue and epilogue
; must not move because a crash can happen anywhere and sanitizers need
; to be able to unwind from the PC of the crash.
; CHECK-LABEL: sanitize:
; CHECK: pushq
; CHECK: incl 40
; CHECK: popq
; CHECK-NEXT: retq
; CHECK: movl $40, %edi
; CHECK-NEXT: callq ___asan_report_load4
define  void @sanitize() #0 {
entry:
  %tmp = load i8, i8* inttoptr (i64 17592186044421 to i8*)
  %tmp1 = icmp ne i8 %tmp, 0
  br i1 %tmp1, label %if.then, label %else

if.then:
  %tmp3 = icmp sge i8 3, %tmp
  br i1 %tmp3, label %else, label %end

else:
  call void @__asan_report_load4(i64 40)
  call void asm sideeffect "", ""()
  unreachable

end:
  %tmp6 = load i32, i32* inttoptr (i64 40 to i32*), align 8
  %inc = add nsw i32 %tmp6, 1
  store i32 %inc, i32* inttoptr (i64 40 to i32*), align 8
  ret void
}

attributes #0 = { sanitize_address nounwind "frame-pointer"="all" }

declare void @__asan_report_load4(i64)
