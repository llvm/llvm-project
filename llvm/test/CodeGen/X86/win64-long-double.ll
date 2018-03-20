; RUN: llc -mtriple x86_64-w64-mingw32 %s -o - | FileCheck %s

@glob = common dso_local local_unnamed_addr global x86_fp80 0xK00000000000000000000, align 16

define dso_local void @call() {
entry:
  %0 = load x86_fp80, x86_fp80* @glob, align 16
  %1 = tail call x86_fp80 @floorl(x86_fp80 %0)
  store x86_fp80 %1, x86_fp80* @glob, align 16
  ret void
}

declare x86_fp80 @floorl(x86_fp80)

; CHECK-LABEL: call
; CHECK: fldt glob(%rip)
; CHECK: fstpt [[ARGOFF:[0-9]+]](%rsp)
; CHECK: leaq [[RETOFF:[0-9]+]](%rsp), %rcx
; CHECK: leaq [[ARGOFF]](%rsp), %rdx
; CHECK: callq floorl
; CHECK: fldt [[RETOFF]](%rsp)
; CHECK: fstpt glob(%rip)
