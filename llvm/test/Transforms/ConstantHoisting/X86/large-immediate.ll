; RUN: opt -mtriple=x86_64-darwin-unknown -S -passes=consthoist < %s | FileCheck %s

define i128 @test1(i128 %a) nounwind {
; CHECK-LABEL: test1
; CHECK: %const = bitcast i128 12297829382473034410122878 to i128
  %1 = add i128 %a, 12297829382473034410122878
  %2 = add i128 %1, 12297829382473034410122878
  ret i128 %2
}

; Check that we don't hoist the shift value of a shift instruction.
define i512 @test2(i512 %a) nounwind {
; CHECK-LABEL: test2
; CHECK-NOT: %const = bitcast i512 504 to i512
  %1 = shl i512 %a, 504
  %2 = ashr i512 %1, 504
  ret i512 %2
}

; Check that we don't hoist constants with a type larger than i128.
define i196 @test3(i196 %a) nounwind {
; CHECK-LABEL: test3
; CHECK-NOT: %const = bitcast i196 2 to i196
  %1 = mul i196 %a, 2
  %2 = mul i196 %1, 2
  ret i196 %2
}

; Check that we don't hoist immediates with small values.
define i96 @test4(i96 %a) nounwind {
; CHECK-LABEL: test4
; CHECK-NOT: %const = bitcast i96 2 to i96
  %1 = mul i96 %a, 2
  %2 = add i96 %1, 2
  ret i96 %2
}
