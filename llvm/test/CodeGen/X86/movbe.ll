; RUN: llc -mtriple=x86_64-linux -mcpu=atom < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-linux -mcpu=slm < %s | FileCheck %s -check-prefix=SLM

declare i16 @llvm.bswap.i16(i16) nounwind readnone
declare i32 @llvm.bswap.i32(i32) nounwind readnone
declare i64 @llvm.bswap.i64(i64) nounwind readnone

define void @test1(ptr nocapture %x, i16 %y) nounwind {
  %bswap = call i16 @llvm.bswap.i16(i16 %y)
  store i16 %bswap, ptr %x, align 2
  ret void
; CHECK-LABEL: test1:
; CHECK: movbew %si, (%rdi)
; SLM-LABEL: test1:
; SLM: movbew   %si, (%rdi)
}

define i16 @test2(ptr %x) nounwind {
  %load = load i16, ptr %x, align 2
  %bswap = call i16 @llvm.bswap.i16(i16 %load)
  ret i16 %bswap
; CHECK-LABEL: test2:
; CHECK: movbew (%rdi), %ax
; SLM-LABEL: test2:
; SLM: movbew   (%rdi), %ax
}

define void @test3(ptr nocapture %x, i32 %y) nounwind {
  %bswap = call i32 @llvm.bswap.i32(i32 %y)
  store i32 %bswap, ptr %x, align 4
  ret void
; CHECK-LABEL: test3:
; CHECK: movbel	%esi, (%rdi)
; SLM-LABEL: test3:
; SLM: movbel	%esi, (%rdi)
}

define i32 @test4(ptr %x) nounwind {
  %load = load i32, ptr %x, align 4
  %bswap = call i32 @llvm.bswap.i32(i32 %load)
  ret i32 %bswap
; CHECK-LABEL: test4:
; CHECK: movbel	(%rdi), %eax
; SLM-LABEL: test4:
; SLM: movbel	(%rdi), %eax
}

define void @test5(ptr %x, i64 %y) nounwind {
  %bswap = call i64 @llvm.bswap.i64(i64 %y)
  store i64 %bswap, ptr %x, align 8
  ret void
; CHECK-LABEL: test5:
; CHECK: movbeq	%rsi, (%rdi)
; SLM-LABEL: test5:
; SLM: movbeq	%rsi, (%rdi)
}

define i64 @test6(ptr %x) nounwind {
  %load = load i64, ptr %x, align 8
  %bswap = call i64 @llvm.bswap.i64(i64 %load)
  ret i64 %bswap
; CHECK-LABEL: test6:
; CHECK: movbeq	(%rdi), %rax
; SLM-LABEL: test6:
; SLM: movbeq	(%rdi), %rax
}
