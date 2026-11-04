; RUN: opt -passes='require<libcall-lowering-info>,safe-stack' -safestack-use-pointer-address -S -mtriple=thumbv7-linux-androideabi < %s | FileCheck %s

; The libcalls use the AAPCS calling convention on both the call and the
; declaration.

; CHECK-LABEL: define void @foo(
; CHECK: call arm_aapcscc ptr @__safestack_pointer_address()
; CHECK: call arm_aapcscc void @__stack_chk_fail()
define void @foo() safestack sspreq {
  %a = alloca [16 x i8]
  call void @Capture(ptr %a)
  ret void
}

declare void @Capture(ptr)

; CHECK: declare arm_aapcscc ptr @__safestack_pointer_address()
; CHECK: declare arm_aapcscc void @__stack_chk_fail()
