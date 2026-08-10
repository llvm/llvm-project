; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi | FileCheck %s

@.str = private unnamed_addr constant [12 x i8] c"val.a = %f\0A\00"
%struct_t = type { double, double, double }
@static_val = constant %struct_t { double 1.0, double 2.0, double 3.0 }

declare i32 @printf(ptr, ...)

; CHECK-LABEL:     test_byval_usage_scheduling:
; CHECK-DAG:   str     r3, [sp, #12]
; CHECK-DAG:   str     r2, [sp, #8]
; CHECK:       vldr    d16, [sp, #8]
define void @test_byval_usage_scheduling(i32 %n1, i32 %n2, ptr byval(%struct_t) %val) nounwind {
entry:
  %0 = load double, ptr %val
  %call = call i32 (ptr, ...) @printf(ptr @.str, double %0)
  ret void
}
