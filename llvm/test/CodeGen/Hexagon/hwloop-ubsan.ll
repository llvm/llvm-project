; RUN: llc -mtriple=hexagon -mcpu=hexagonv68 -O2 < %s | FileCheck %s

; The loop has a normal, data-dependent exit and a separate pointer-overflow
; sanitizer exit.  The latter must not be used as the hardware-loop count.

; CHECK-LABEL: strlen:
; CHECK-NOT: loop0
; CHECK: memb
; CHECK: jumpr

define i32 @strlen(ptr %str) {
entry:
  %0 = ptrtoint ptr %str to i32
  %1 = xor i32 %0, -1
  br label %loop

loop:
  %len = phi i32 [ 0, %entry ], [ %inc, %body ]
  %ptr = getelementptr inbounds nuw i8, ptr %str, i32 %len
  %ch = load i8, ptr %ptr, align 1
  %end = icmp eq i8 %ch, 0
  br i1 %end, label %done, label %body

body:
  call void asm sideeffect "", ""()
  %inc = add i32 %len, 1
  %overflow = icmp ugt i32 %inc, %1
  br i1 %overflow, label %trap, label %loop

trap:
  call void @llvm.ubsantrap(i8 19)
  unreachable

done:
  ret i32 %len
}

declare void @llvm.ubsantrap(i8 immarg)
