; RUN: opt -mtriple=riscv32-unknown-elf -S -consthoist < %s | FileCheck %s
; RUN: opt -mtriple=riscv64-unknown-elf -S -consthoist < %s | FileCheck %s

; Check that we don't hoist immediates with small values.
define i64 @test1(i64 %a) nounwind {
; CHECK-LABEL: test1
; CHECK-NOT: %const = bitcast i64 2 to i64
  %1 = mul i64 %a, 2
  %2 = add i64 %1, 2
  ret i64 %2
}

; Check that we don't hoist immediates with small values.
define i64 @test2(i64 %a) nounwind {
; CHECK-LABEL: test2
; CHECK-NOT: %const = bitcast i64 2047 to i64
  %1 = mul i64 %a, 2047
  %2 = add i64 %1, 2047
  ret i64 %2
}

; Check that we hoist immediates with large values.
define i64 @test3(i64 %a) nounwind {
; CHECK-LABEL: test3
; CHECK: %const = bitcast i64 32767 to i64
  %1 = mul i64 %a, 32767
  %2 = add i64 %1, 32767
  ret i64 %2
}