; RUN: llvm-as < %s > %t.bc
; RUN: llvm-lto -O0 -relocation-model=static -o %t.o %t.bc
; RUN: llvm-objdump -d -r %t.o | FileCheck %s

; Confirm that we do generate one too many indirections accessing the stack guard
; variable, when the relocation model is static and the PIC level is not 0..
; This is preparation for the fix.
;
target triple = "armv4t-unknown-unknown"

define arm_aapcscc i8 @foo() #0 {
entry:
  %arr = alloca [200 x i8], align 1
  call void @llvm.memset.p0.i32(ptr align 1 %arr, i8 0, i32 200, i1 false)
  %arrayidx = getelementptr inbounds [200 x i8], ptr %arr, i32 0, i8 5
  %0 = load i8, ptr %arrayidx, align 1
  ret i8 %0
}

; CHECK:      <foo>:
; CHECK:      [[#%x,CURPC:]]:{{.*}} ldr r[[REG1:[0-9]+]], [pc, #0x[[#%x,OFFSET:]]]
; CHECK-NEXT: ldr r[[REG2:[0-9]+]], [r[[REG1]]]
; CHECK-NEXT: ldr r[[REG3:[0-9]+]], [r[[REG2]]]
; CHECK-NEXT: str r[[REG3]],
; CHECK:      [[#CURPC + OFFSET + 8]]:{{.*}}.word
; CHECK-NEXT: R_ARM_ABS32 __stack_chk_guard

declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1 immarg)

define arm_aapcscc i32 @main() {
entry:
  %call = call arm_aapcscc i8 @foo()
  %conv = zext i8 %call to i32
  ret i32 %conv
}

attributes #0 = { sspstrong }

!llvm.module.flags = !{!0}
!0 = !{i32 8, !"PIC Level", i32 2}
