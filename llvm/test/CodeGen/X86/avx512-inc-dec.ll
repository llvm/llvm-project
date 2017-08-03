; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s

;CHECK-LABEL: test
;CHECK-NOT: dec
;CHECK-NOT: enc
;CHECK: ret
define i32 @test(i32 %a, i32 %b) {
 %a1 = add i32 %a, -1
 %b1 = add i32 %b, 1
 %res = mul i32 %a1, %b1
 ret i32 %res
}

