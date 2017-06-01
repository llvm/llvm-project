; Testcase generated from the following code:
; extern __thread int i;
; void f();
; int g(void) {
;   if (i) {
;     i = 0;
;     f();
;   }
;   return i;
; }
; We want to make sure that TLS variables are not accessed before
; the stack frame is set up.

; RUN: llc < %s -relocation-model=pic | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-freebsd11.0"

@i = external thread_local global i32, align 4

define i32 @g() #0 {
entry:
  %tmp = load i32, i32* @i, align 4
  %tobool = icmp eq i32 %tmp, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  store i32 0, i32* @i, align 4
  tail call void (...) @f() #2
  %.pre = load i32, i32* @i, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %tmp1 = phi i32 [ 0, %entry ], [ %.pre, %if.then ]
  ret i32 %tmp1
}

; CHECK: g:                                      # @g
; CHECK-NEXT:         .cfi_startproc
; CHECK-NEXT: # BB#0:                                 # %entry
; CHECK-NEXT:         pushq   %rbp
; CHECK-NEXT: .Lcfi0:
; CHECK-NEXT:         .cfi_def_cfa_offset 16
; CHECK-NEXT: .Lcfi1:
; CHECK-NEXT:         .cfi_offset %rbp, -16
; CHECK-NEXT:         movq    %rsp, %rbp
; CHECK-NEXT: .Lcfi2:
; CHECK-NEXT:         .cfi_def_cfa_register %rbp
; CHECK-NEXT:         pushq   %rbx
; CHECK-NEXT:         pushq   %rax
; CHECK-NEXT: .Lcfi3:
; CHECK-NEXT:         .cfi_offset %rbx, -24
; CHECK-NEXT:         data16
; CHECK-NEXT:         leaq    i@TLSGD(%rip), %rdi

declare void @f(...) #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
