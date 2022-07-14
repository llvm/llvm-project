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

; Check that we hoist immediates with very large values.
define i128 @test4(i128 %a) nounwind {
; CHECK-LABEL: test4
; CHECK: %const = bitcast i128 12297829382473034410122878 to i128
  %1 = add i128 %a, 12297829382473034410122878
  %2 = add i128 %1, 12297829382473034410122878
  ret i128 %2
}

; Check that we hoist zext.h without Zbb.
define i32 @test5(i32 %a) nounwind {
; CHECK-LABEL: test5
; CHECK: %const = bitcast i32 65535 to i32
  %1 = and i32 %a, 65535
  %2 = and i32 %1, 65535
  ret i32 %2
}

; Check that we don't hoist zext.h with 65535 with Zbb.
define i32 @test6(i32 %a) nounwind "target-features"="+zbb" {
; CHECK-LABEL: test6
; CHECK: and i32 %a, 65535
  %1 = and i32 %a, 65535
  %2 = and i32 %1, 65535
  ret i32 %2
}

; Check that we hoist zext.w without Zba.
define i64 @test7(i64 %a) nounwind {
; CHECK-LABEL: test7
; CHECK: %const = bitcast i64 4294967295 to i64
  %1 = and i64 %a, 4294967295
  %2 = and i64 %1, 4294967295
  ret i64 %2
}

; Check that we don't hoist zext.w with Zba.
define i64 @test8(i64 %a) nounwind "target-features"="+zba" {
; CHECK-LABEL: test8
; CHECK: and i64 %a, 4294967295
  %1 = and i64 %a, 4294967295
  %2 = and i64 %1, 4294967295
  ret i64 %2
}
