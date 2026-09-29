; RUN: llvm-reduce --delta-passes=simplify-cfg --test %python --test-arg %p/Inputs/remove-bbs.py -abort-on-invalid-reduction %s -o %t

; RUN: FileCheck --check-prefix=CHECK-FINAL %s --input-file=%t
; CHECK-FINAL: @f1
; CHECK-FINAL-NOT: x6:
; CHECK-FINAL-NOT: x10:

define void @f1(ptr %interesting3, i32 %interesting2) {
  %x3 = alloca ptr, i32 0, align 8
  store ptr %interesting3, ptr %interesting3, align 8
  switch i32 %interesting2, label %interesting1 [
    i32 0, label %x6
    i32 1, label %x11
  ]

x4:
  %x5 = call ptr @f2()
  br label %x10

x10:
  br label %interesting1

x6:
  br label %x11

x11:
  br label %interesting1

interesting1:
  ret void
}

declare ptr @f2()
