; RUN: opt < %s -passes=instcombine -S | FileCheck %s

; crc32 with 64-bit destination zeros high 32-bit.
; rdar://9467055

define i64 @test() nounwind {
entry:
; CHECK: test
; CHECK: tail call i64 @llvm.x86.sse42.crc32.64.64
; CHECK-NOT: and
; CHECK: ret
  %0 = tail call i64 @llvm.x86.sse42.crc32.64.64(i64 0, i64 4) nounwind
  %1 = and i64 %0, 4294967295
  ret i64 %1
}

declare i64 @llvm.x86.sse42.crc32.64.64(i64, i64) nounwind readnone
