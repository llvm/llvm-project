; RUN: %lli -jit-kind=mcjit -force-interpreter=true %s | FileCheck %s
; CHECK: int test passed
; CHECK: double test passed
; CHECK: float test passed

@msg_int = internal global [17 x i8] c"int test passed\0A\00"
@msg_double = internal global [20 x i8] c"double test passed\0A\00"
@msg_float = internal global [19 x i8] c"float test passed\0A\00"

declare i32 @printf(ptr, ...)

define i32 @main() {
  %a = alloca <4 x i32>, align 16
  %b = alloca <4 x double>, align 16
  %c = alloca <4 x float>, align 16
  %pint_0 = alloca i32
  %pint_1 = alloca i32
  %pint_2 = alloca i32
  %pint_3 = alloca i32
  %pdouble_0 = alloca double
  %pdouble_1 = alloca double
  %pdouble_2 = alloca double
  %pdouble_3 = alloca double
  %pfloat_0 = alloca float
  %pfloat_1 = alloca float
  %pfloat_2 = alloca float
  %pfloat_3 = alloca float

  ; store constants 1,2,3,4 as vector
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %a, align 16
  ; store constants 1,2,3,4 as scalars
  store i32 1, ptr %pint_0
  store i32 2, ptr %pint_1
  store i32 3, ptr %pint_2
  store i32 4, ptr %pint_3
  
  ; load stored scalars
  %val_int0 = load i32, ptr %pint_0
  %val_int1 = load i32, ptr %pint_1
  %val_int2 = load i32, ptr %pint_2
  %val_int3 = load i32, ptr %pint_3

  ; load stored vector
  %val0 = load <4 x i32> , ptr %a, align 16

  ; extract integers from the loaded vector
  %res_i32_0 = extractelement <4 x i32> %val0, i32 0
  %res_i32_1 = extractelement <4 x i32> %val0, i32 1
  %res_i32_2 = extractelement <4 x i32> %val0, i32 2
  %res_i32_3 = extractelement <4 x i32> %val0, i32 3

  ; compare extracted data with stored constants
  %test_result_int_0 = icmp eq i32 %res_i32_0, %val_int0
  %test_result_int_1 = icmp eq i32 %res_i32_1, %val_int1
  %test_result_int_2 = icmp eq i32 %res_i32_2, %val_int2
  %test_result_int_3 = icmp eq i32 %res_i32_3, %val_int3

  %test_result_int_4 = icmp eq i32 %res_i32_0, %val_int3
  %test_result_int_5 = icmp eq i32 %res_i32_1, %val_int2
  %test_result_int_6 = icmp eq i32 %res_i32_2, %val_int1
  %test_result_int_7 = icmp eq i32 %res_i32_3, %val_int0

  ; it should be TRUE
  %A_i = or i1 %test_result_int_0, %test_result_int_4
  %B_i = or i1 %test_result_int_1, %test_result_int_5
  %C_i = or i1 %test_result_int_2, %test_result_int_6
  %D_i = or i1 %test_result_int_3, %test_result_int_7
  %E_i = and i1 %A_i, %B_i
  %F_i = and i1 %C_i, %D_i
  %res_i = and i1 %E_i, %F_i

  ; if TRUE print message
  br i1 %res_i, label %Print_int, label %Double
Print_int:
  call i32 (ptr,...) @printf(ptr @msg_int)
  br label %Double
Double:
  store <4 x double> <double 5.0, double 6.0, double 7.0, double 8.0>, ptr %b, align 16
  ; store constants as scalars
  store double 5.0, ptr %pdouble_0
  store double 6.0, ptr %pdouble_1
  store double 7.0, ptr %pdouble_2
  store double 8.0, ptr %pdouble_3

  ; load stored vector
  %val1 = load <4 x double> , ptr %b, align 16
  ; load stored scalars
  %val_double0 = load double, ptr %pdouble_0
  %val_double1 = load double, ptr %pdouble_1
  %val_double2 = load double, ptr %pdouble_2
  %val_double3 = load double, ptr %pdouble_3

  %res_double_0 = extractelement <4 x double> %val1, i32 0
  %res_double_1 = extractelement <4 x double> %val1, i32 1
  %res_double_2 = extractelement <4 x double> %val1, i32 2
  %res_double_3 = extractelement <4 x double> %val1, i32 3

  %test_result_double_0 = fcmp oeq double %res_double_0, %val_double0
  %test_result_double_1 = fcmp oeq double %res_double_1, %val_double1
  %test_result_double_2 = fcmp oeq double %res_double_2, %val_double2
  %test_result_double_3 = fcmp oeq double %res_double_3, %val_double3

  %test_result_double_4 = fcmp oeq double %res_double_0, %val_double3
  %test_result_double_5 = fcmp oeq double %res_double_1, %val_double2
  %test_result_double_6 = fcmp oeq double %res_double_2, %val_double1
  %test_result_double_7 = fcmp oeq double %res_double_3, %val_double0

  %A_double = or i1 %test_result_double_0, %test_result_double_4
  %B_double = or i1 %test_result_double_1, %test_result_double_5
  %C_double = or i1 %test_result_double_2, %test_result_double_6
  %D_double = or i1 %test_result_double_3, %test_result_double_7
  %E_double = and i1 %A_double, %B_double
  %F_double = and i1 %C_double, %D_double
  %res_double = and i1 %E_double, %F_double

  br i1 %res_double, label %Print_double, label %Float
Print_double:
  call i32 (ptr,...) @printf(ptr @msg_double)
  br label %Float
Float:
  store <4 x float> <float 9.0, float 10.0, float 11.0, float 12.0>, ptr %c, align 16

  store float 9.0, ptr %pfloat_0
  store float 10.0, ptr %pfloat_1
  store float 11.0, ptr %pfloat_2
  store float 12.0, ptr %pfloat_3

  ; load stored vector
  %val2 = load <4 x float> , ptr %c, align 16
  ; load stored scalars
  %val_float0 = load float, ptr %pfloat_0
  %val_float1 = load float, ptr %pfloat_1
  %val_float2 = load float, ptr %pfloat_2
  %val_float3 = load float, ptr %pfloat_3

  %res_float_0 = extractelement <4 x float> %val2, i32 0
  %res_float_1 = extractelement <4 x float> %val2, i32 1
  %res_float_2 = extractelement <4 x float> %val2, i32 2
  %res_float_3 = extractelement <4 x float> %val2, i32 3

  %test_result_float_0 = fcmp oeq float %res_float_0, %val_float0
  %test_result_float_1 = fcmp oeq float %res_float_1, %val_float1
  %test_result_float_2 = fcmp oeq float %res_float_2, %val_float2
  %test_result_float_3 = fcmp oeq float %res_float_3, %val_float3

  %test_result_float_4 = fcmp oeq float %res_float_0, %val_float3
  %test_result_float_5 = fcmp oeq float %res_float_1, %val_float2
  %test_result_float_6 = fcmp oeq float %res_float_2, %val_float1
  %test_result_float_7 = fcmp oeq float %res_float_3, %val_float0

  %A_float = or i1 %test_result_float_0, %test_result_float_4
  %B_float = or i1 %test_result_float_1, %test_result_float_5
  %C_float = or i1 %test_result_float_2, %test_result_float_6
  %D_float = or i1 %test_result_float_3, %test_result_float_7
  %E_float = and i1 %A_float, %B_float
  %F_float = and i1 %C_float, %D_float
  %res_float = and i1 %E_float, %F_float

  br i1 %res_float, label %Print_float, label %Exit
Print_float:
  call i32 (ptr,...) @printf(ptr @msg_float)
  br label %Exit
Exit:

  ret i32 0
}
