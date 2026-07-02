; RUN: llc < %s -mtriple=ve | FileCheck %s

; Verify that @llvm.eh.sjlj.setjmp stores FP and SP into the buffer.

@buf = common global [1 x [25 x i64]] zeroinitializer, align 8

declare i32 @llvm.eh.sjlj.setjmp(ptr) nounwind

define i32 @setjmp_test() nounwind "frame-pointer"="all" {
  %r = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret i32 %r
}

; CHECK-LABEL: setjmp_test:
; CHECK:       st %s9, (, %s0)
; CHECK:       st %s11, 16(, %s0)
