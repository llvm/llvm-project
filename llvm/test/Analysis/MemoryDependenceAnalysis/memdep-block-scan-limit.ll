; RUN: opt -S -gvn < %s | FileCheck %s
; RUN: opt -S -memdep-block-scan-limit=1 -gvn < %s | FileCheck %s --check-prefix=WITH-LIMIT
; CHECK-LABEL: @test(
; CHECK: load
; CHECK-NOT: load
; WITH-LIMIT-LABEL: @test(
; WITH-LIMIT-CHECK: load
; WITH-LIMIT-CHECK: load
define i32 @test(i32* %p) {
 %1 = load i32, i32* %p
 %2 = add i32 %1, 3
 %3 = load i32, i32* %p
 %4 = add i32 %2, %3
 ret i32 %4
}
