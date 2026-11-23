; RUN: llc -mtriple=x86_64-pc-windows-msvc -verify-machineinstrs < %s | FileCheck %s

; We should store -2 into UnwindHelp in a slot immediately after the last XMM
; CSR save.

declare void @g()
declare i32 @__CxxFrameHandler3(...)

@fp_global = global double 0.0

define void @f() personality ptr @__CxxFrameHandler3 {
  %v = load double, ptr @fp_global
  call void @g()
  %v1 = fadd double %v, 1.0
  store double %v1, ptr @fp_global
  invoke void @g()
      to label %return unwind label %catch.dispatch

return:
  ret void

catch.dispatch:
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:
  %p = catchpad within %cs1 [ptr null, i32 64, ptr null]
  catchret from %p to label %return
}

; CHECK: f: # @f
; CHECK: pushq   %rbp
; CHECK: .seh_pushreg %rbp
; CHECK: subq    $64, %rsp
; CHECK: .seh_stackalloc 64
; CHECK: leaq    64(%rsp), %rbp
; CHECK: .seh_setframe %rbp, 64
; CHECK: movaps  %xmm6, -16(%rbp)        # 16-byte Spill
; CHECK: .seh_savexmm %xmm6, 48
; CHECK: .seh_endprologue
; CHECK: movq    $-2, -24(%rbp)
; CHECK: movsd   fp_global(%rip), %xmm6  # xmm6 = mem[0],zero
; CHECK: callq   g
; CHECK: addsd   __real@3ff0000000000000(%rip), %xmm6
; CHECK: movsd   %xmm6, fp_global(%rip)
; CHECK: .Ltmp{{.*}}
; CHECK: callq   g
; CHECK: .Ltmp{{.*}}
; CHECK: .LBB{{.*}} # Block address taken
; CHECK: movaps  -16(%rbp), %xmm6
; CHECK: .seh_startepilogue
; CHECK: addq    $64, %rsp
; CHECK: popq    %rbp
; CHECK: .seh_endepilogue
; CHECK: retq
; CHECK: .seh_handlerdata
; CHECK: # %catch
; CHECK: movq    %rdx, 16(%rsp)
; CHECK: pushq   %rbp
; CHECK: .seh_pushreg %rbp
; CHECK: subq    $48, %rsp
; CHECK: .seh_stackalloc 48
; CHECK: leaq    64(%rdx), %rbp
; CHECK: movapd  %xmm6, 32(%rsp)
; CHECK: .seh_savexmm %xmm6, 32
; CHECK: .seh_endprologue
; CHECK: movapd  32(%rsp), %xmm6
; CHECK: leaq    .LBB0_1(%rip), %rax
; CHECK: .seh_startepilogue
; CHECK: addq    $48, %rsp
; CHECK: popq    %rbp
; CHECK: .seh_endepilogue
; CHECK: retq # CATCHRET
