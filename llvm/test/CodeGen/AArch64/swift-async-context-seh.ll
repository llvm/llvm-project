; RUN: rm -rf %t && mkdir -p %t
; RUN: llc -mtriple aarch64-unknown-windows-msvc %s -o - | FileCheck %s
; RUN: llc -mtriple aarch64-unknown-windows-msvc -filetype obj %s -o %t/a.o

; Check that the prologue/epilogue instructions for the swift async
; context have an associated SEH instruction and that it doesn't error
; when the output is an object file.

; CHECK: orr     x29, x29, #0x1000000000000000
; CHECK-NEXT: .seh_nop
; CHECK:  str     x22, [sp, #16]
; CHECK-NEXT: .seh_nop
; CHECK: and     x29, x29, #0xefffffffffffffff
; CHECK-NEXT: .seh_nop

declare ptr @llvm.swift.async.context.addr()

define internal swifttailcc void @test(ptr nocapture readonly swiftasync %0) {
entryresume.0:
  %1 = load ptr, ptr %0, align 8
  %2 = tail call ptr @llvm.swift.async.context.addr()
  store ptr %1, ptr %2, align 8
  ret void
}
