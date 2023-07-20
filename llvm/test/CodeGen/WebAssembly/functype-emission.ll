; RUN: llc -mtriple=wasm32 -verify-machineinstrs < %s | FileCheck %s

; Demonstrates that appropriate .functype directives are emitted for defined
; functions, declared functions, and any libcalls.

; CHECK: .functype __multi3 (i32, i64, i64, i64, i64) -> ()
; CHECK: .functype __unordtf2 (i64, i64, i64, i64) -> (i32)
; CHECK: .functype defined_fun_1 (f64) -> (i64)
; CHECK: .functype defined_fun_2 (f64, i32) -> (i64)
; CHECK: .functype declared_fun (i32, f32, i64) -> (i32)

define i64 @defined_fun_1(double %a) {
; CHECK-LABEL: defined_fun_1:
; CHECK:         .functype defined_fun_1 (f64) -> (i64)
  %1 = call i64 @defined_fun_2(double %a, i32 1)
  ret i64 %1
}

define i64 @defined_fun_2(double %a, i32 %b) {
; CHECK-LABEL: defined_fun_2:
; CHECK:         .functype defined_fun_2 (f64, i32) -> (i64)
  %1 = call i64 @defined_fun_1(double %a)
  ret i64 %1
}

declare i8 @declared_fun(i32, float, i64)

define i128 @libcalls(fp128 %a, fp128 %b, i128 %c, i128 %d) {
; CHECK-LABEL: libcalls:
; CHECK:         .functype libcalls (i32, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
 %1 = fcmp uno fp128 %a, %b
 %2 = zext i1 %1 to i128
 %3 = mul i128 %c, %d
 %4 = add i128 %2, %3
 ret i128 %4
}
