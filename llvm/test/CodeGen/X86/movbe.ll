; RUN: llc -mtriple=x86_64-linux -mcpu=atom < %s | FileCheck %s

declare i32 @llvm.bswap.i32(i32) nounwind readnone
declare i64 @llvm.bswap.i64(i64) nounwind readnone

define void @test1(i32* nocapture %x, i32 %y) nounwind {
  %bswap = call i32 @llvm.bswap.i32(i32 %y)
  store i32 %bswap, i32* %x, align 4
  ret void
; CHECK: test1:
; CHECK: movbel	%esi, (%rdi)
}

define i32 @test2(i32* %x) nounwind {
  %load = load i32* %x, align 4
  %bswap = call i32 @llvm.bswap.i32(i32 %load)
  ret i32 %bswap
; CHECK: test2:
; CHECK: movbel	(%rdi), %eax
}

define void @test3(i64* %x, i64 %y) nounwind {
  %bswap = call i64 @llvm.bswap.i64(i64 %y)
  store i64 %bswap, i64* %x, align 8
  ret void
; CHECK: test3:
; CHECK: movbeq	%rsi, (%rdi)
}

define i64 @test4(i64* %x) nounwind {
  %load = load i64* %x, align 8
  %bswap = call i64 @llvm.bswap.i64(i64 %load)
  ret i64 %bswap
; CHECK: test4:
; CHECK: movbeq	(%rdi), %rax
}
