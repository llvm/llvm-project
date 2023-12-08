; RUN: opt -mtriple=arm64-darwin-unknown -S -passes=consthoist < %s | FileCheck %s

define i128 @test1(i128 %a) nounwind {
; CHECK-LABEL: test1
; CHECK: %const = bitcast i128 12297829382473034410122878 to i128
  %1 = add i128 %a, 12297829382473034410122878
  %2 = add i128 %1, 12297829382473034410122878
  ret i128 %2
}

; Check that we don't hoist large, but cheap constants
define i512 @test2(i512 %a) nounwind {
; CHECK-LABEL: test2
; CHECK-NOT: %const = bitcast i512 7 to i512
  %1 = and i512 %a, 7
  %2 = or i512 %1, 7
  ret i512 %2
}

; Check that we don't hoist the shift value of a shift instruction.
define i512 @test3(i512 %a) nounwind {
; CHECK-LABEL: test3
; CHECK-NOT: %const = bitcast i512 504 to i512
  %1 = shl i512 %a, 504
  %2 = ashr i512 %1, 504
  ret i512 %2
}
