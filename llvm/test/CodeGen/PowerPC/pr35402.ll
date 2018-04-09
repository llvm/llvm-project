; RUN: llc -O2 < %s | FileCheck %s
target triple = "powerpc64le-linux-gnu"

define void @test(i8* %p, i64 %data) {
entry:
  %0 = tail call i64 @llvm.bswap.i64(i64 %data)
  %ptr = bitcast i8* %p to i48*
  %val = trunc i64 %0 to i48
  store i48 %val, i48* %ptr, align 1
  ret void

; CHECK:     sth
; CHECK:     stw
; CHECK-NOT: stdbrx

}

declare i64 @llvm.bswap.i64(i64)
