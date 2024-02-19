; RUN: llc -mtriple aarch64-unknown-windows-msvc -filetype asm %s -o - | FileCheck %s

; Check that the slot offset of the async context (x22) doesn't
; conflict with that of another callee-saved register (x21 here) and
; saving it won't overwrite the saved value of the callee-saved
; register.
;
; CHECK:        sub     sp, sp, #64
; CHECK:        str     x19, [sp, #16]
; CHECK:        str     x21, [sp, #24]
; CHECK-NOT:    stp     x29, x30, [sp, #32]
; CHECK:        stp     x29, x30, [sp, #40]
; CHECK-NOT:    str     x22, [sp, #24]
; CHECK:        str     x22, [sp, #32]

declare ptr @llvm.swift.async.context.addr()
declare swiftcc i64 @foo(i64 %0, i64 %1)
declare swifttailcc void @tail(ptr swiftasync %0, ptr swiftself dereferenceable(8) %1, i64 %2)
define internal swifttailcc void @test(ptr swiftasync %0, ptr swiftself %1, i64 %2) {
entry:
  %3 = load ptr, ptr %0, align 8
  %4 = call ptr @llvm.swift.async.context.addr()
  store ptr %3, ptr %4, align 8
  %5 = call swiftcc i64 @foo(i64 %2, i64 %2)
  %6 = call swiftcc i64 @foo(i64 %2, i64 %5)
  %7 = call swiftcc i64 @foo(i64 %5, i64 %2)
  %8 = call swiftcc i64 @foo(i64 %7, i64 %6)
  %9 = call swiftcc i64 @foo(i64 %2, i64 %8)
  %10 = call ptr @llvm.swift.async.context.addr()
  musttail call swifttailcc void @tail(ptr swiftasync %10, ptr swiftself %1, i64 %2)
  ret void
}
