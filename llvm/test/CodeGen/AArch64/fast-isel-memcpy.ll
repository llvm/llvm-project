; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s

; Test that we don't segfault.
; CHECK-LABEL: test
; CHECK:       ldr [[REG1:x[0-9]+]], [x1]
; CHECK-NEXT:  and [[REG2:x[0-9]+]], x0, #0x7fffffffffffffff
; CHECK-NEXT:  str [[REG1]], {{\[}}[[REG2]]{{\]}}
define void @test(i64 %a, i8* %b) {
  %1 = and i64 %a, 9223372036854775807
  %2 = inttoptr i64 %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %2, i8* %b, i64 8, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)
