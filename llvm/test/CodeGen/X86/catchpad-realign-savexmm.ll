; RUN: llc -mtriple=x86_64-pc-windows-msvc -verify-machineinstrs < %s | FileCheck %s

; We should store -2 into UnwindHelp in a slot immediately after the last XMM
; CSR save.

declare void @g()
declare i32 @__CxxFrameHandler3(...)

@fp_global = global double 0.0

define void @f() personality i32 (...)* @__CxxFrameHandler3 {
  %v = load double, double* @fp_global
  call void @g()
  %v1 = fadd double %v, 1.0
  store double %v1, double* @fp_global
  invoke void @g()
      to label %return unwind label %catch

return:
  ret void

catch:
  %p = catchpad [i8* null, i32 64, i8* null]
      to label %catchit unwind label %endpad

catchit:
  catchret %p to label %return
endpad:
  catchendpad unwind to caller
}

; CHECK: f: # @f
; CHECK: pushq   %rbp
; CHECK: .seh_pushreg 5
; CHECK: subq    $64, %rsp
; CHECK: .seh_stackalloc 64
; CHECK: leaq    64(%rsp), %rbp
; CHECK: .seh_setframe 5, 64
; CHECK: movaps  %xmm6, -16(%rbp)        # 16-byte Spill
; CHECK: .seh_savexmm 6, 48
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
; CHECK: addq    $64, %rsp
; CHECK: popq    %rbp
; CHECK: retq
; CHECK: .seh_handlerdata
