; RUN: opt -instcombine -o - -S %s | FileCheck %s

; The constant at %v35 should be shrunk, but this must lead to the nsw flag of
; %v43 getting removed so that %v44 is not illegally optimized away.
; CHECK-LABEL: @foo
; CHECK: %v35 = add nuw nsw i32 %v34, 1362915575
; ...
; CHECK: add nuw i32 %v42, 1533579450
; CHECK-NEXT: %v44 = or i32 %v43, -2147483648
; CHECK-NEXT: %v45 = xor i32 %v44, 749011377
; CHECK-NEXT: ret i32 %v45
define i32 @foo(i32 %arg) {
  %v33 = and i32 %arg, 223
  %v34 = xor i32 %v33, 29
  %v35 = add nuw i32 %v34, 3510399223
  %v37 = or i32 %v34, 1874836915
  %v38 = and i32 %v34, 221
  %v39 = xor i32 %v38, 1874836915
  %v40 = xor i32 %v37, %v39
  %v41 = shl nsw nuw i32 %v40, 1
  %v42 = sub i32 %v35, %v41
  %v43 = add nsw i32 %v42, 1533579450
  %v44 = or i32 %v43, -2147483648
  %v45 = xor i32 %v44, 749011377
  ret i32 %v45
}
