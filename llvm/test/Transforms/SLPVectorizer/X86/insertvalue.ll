; RUN: opt < %s -basicaa -slp-vectorizer -S -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7-avx | FileCheck %s

; CHECK-LABEL: julia_2xdouble
; CHECK: load <2 x double>
; CHECK: load <2 x double>
; CHECK: fmul <2 x double>
; CHECK: fadd <2 x double>
define void @julia_2xdouble([2 x double]* sret, [2 x double]*, [2 x double]*, [2 x double]*) {
top:
  %px0 = getelementptr inbounds [2 x double], [2 x double]* %2, i64 0, i64 0
  %x0 = load double, double* %px0, align 4
  %py0 = getelementptr inbounds [2 x double], [2 x double]* %3, i64 0, i64 0
  %y0 = load double, double* %py0, align 4
  %m0 = fmul double %x0, %y0
  %px1 = getelementptr inbounds [2 x double], [2 x double]* %2, i64 0, i64 1
  %x1 = load double, double* %px1, align 4
  %py1 = getelementptr inbounds [2 x double], [2 x double]* %3, i64 0, i64 1
  %y1 = load double, double* %py1, align 4
  %m1 = fmul double %x1, %y1
  %pz0 = getelementptr inbounds [2 x double], [2 x double]* %1, i64 0, i64 0
  %z0 = load double, double* %pz0, align 4
  %a0 = fadd double %m0, %z0
  %i0 = insertvalue [2 x double] undef, double %a0, 0
  %pz1 = getelementptr inbounds [2 x double], [2 x double]* %1, i64 0, i64 1
  %z1 = load double, double* %pz1, align 4
  %a1 = fadd double %m1, %z1
  %i1 = insertvalue [2 x double] %i0, double %a1, 1
  store [2 x double] %i1, [2 x double]* %0, align 4
  ret void
}

; CHECK-LABEL: julia_4xfloat
; CHECK: load <4 x float>
; CHECK: load <4 x float>
; CHECK: fmul <4 x float>
; CHECK: fadd <4 x float>
define void @julia_4xfloat([4 x float]* sret, [4 x float]*, [4 x float]*, [4 x float]*) {
top:
  %px0 = getelementptr inbounds [4 x float], [4 x float]* %2, i64 0, i64 0
  %x0 = load float, float* %px0, align 4
  %py0 = getelementptr inbounds [4 x float], [4 x float]* %3, i64 0, i64 0
  %y0 = load float, float* %py0, align 4
  %m0 = fmul float %x0, %y0
  %px1 = getelementptr inbounds [4 x float], [4 x float]* %2, i64 0, i64 1
  %x1 = load float, float* %px1, align 4
  %py1 = getelementptr inbounds [4 x float], [4 x float]* %3, i64 0, i64 1
  %y1 = load float, float* %py1, align 4
  %m1 = fmul float %x1, %y1
  %px2 = getelementptr inbounds [4 x float], [4 x float]* %2, i64 0, i64 2
  %x2 = load float, float* %px2, align 4
  %py2 = getelementptr inbounds [4 x float], [4 x float]* %3, i64 0, i64 2
  %y2 = load float, float* %py2, align 4
  %m2 = fmul float %x2, %y2
  %px3 = getelementptr inbounds [4 x float], [4 x float]* %2, i64 0, i64 3
  %x3 = load float, float* %px3, align 4
  %py3 = getelementptr inbounds [4 x float], [4 x float]* %3, i64 0, i64 3
  %y3 = load float, float* %py3, align 4
  %m3 = fmul float %x3, %y3
  %pz0 = getelementptr inbounds [4 x float], [4 x float]* %1, i64 0, i64 0
  %z0 = load float, float* %pz0, align 4
  %a0 = fadd float %m0, %z0
  %i0 = insertvalue [4 x float] undef, float %a0, 0
  %pz1 = getelementptr inbounds [4 x float], [4 x float]* %1, i64 0, i64 1
  %z1 = load float, float* %pz1, align 4
  %a1 = fadd float %m1, %z1
  %i1 = insertvalue [4 x float] %i0, float %a1, 1
  %pz2 = getelementptr inbounds [4 x float], [4 x float]* %1, i64 0, i64 2
  %z2 = load float, float* %pz2, align 4
  %a2 = fadd float %m2, %z2
  %i2 = insertvalue [4 x float] %i1, float %a2, 2
  %pz3 = getelementptr inbounds [4 x float], [4 x float]* %1, i64 0, i64 3
  %z3 = load float, float* %pz3, align 4
  %a3 = fadd float %m3, %z3
  %i3 = insertvalue [4 x float] %i2, float %a3, 3
  store [4 x float] %i3, [4 x float]* %0, align 4
  ret void
}

; CHECK-LABEL: julia_load_array_of_float
; CHECK: fsub <4 x float>
define void @julia_load_array_of_float([4 x float]* %a, [4 x float]* %b, [4 x float]* %c) {
top:
  %a_arr = load [4 x float], [4 x float]* %a, align 4
  %a0 = extractvalue [4 x float] %a_arr, 0
  %a2 = extractvalue [4 x float] %a_arr, 2
  %a1 = extractvalue [4 x float] %a_arr, 1
  %b_arr = load [4 x float], [4 x float]* %b, align 4
  %b0 = extractvalue [4 x float] %b_arr, 0
  %b2 = extractvalue [4 x float] %b_arr, 2
  %b1 = extractvalue [4 x float] %b_arr, 1
  %a3 = extractvalue [4 x float] %a_arr, 3
  %c1 = fsub float %a1, %b1
  %b3 = extractvalue [4 x float] %b_arr, 3
  %c0 = fsub float %a0, %b0
  %c2 = fsub float %a2, %b2
  %c_arr0 = insertvalue [4 x float] undef, float %c0, 0
  %c_arr1 = insertvalue [4 x float] %c_arr0, float %c1, 1
  %c3 = fsub float %a3, %b3
  %c_arr2 = insertvalue [4 x float] %c_arr1, float %c2, 2
  %c_arr3 = insertvalue [4 x float] %c_arr2, float %c3, 3
  store [4 x float] %c_arr3, [4 x float]* %c, align 4
  ret void
}

; CHECK-LABEL: julia_load_array_of_i32
; CHECK: load <4 x i32>
; CHECK: load <4 x i32>
; CHECK: sub <4 x i32>
define void @julia_load_array_of_i32([4 x i32]* %a, [4 x i32]* %b, [4 x i32]* %c) {
top:
  %a_arr = load [4 x i32], [4 x i32]* %a, align 4
  %a0 = extractvalue [4 x i32] %a_arr, 0
  %a2 = extractvalue [4 x i32] %a_arr, 2
  %a1 = extractvalue [4 x i32] %a_arr, 1
  %b_arr = load [4 x i32], [4 x i32]* %b, align 4
  %b0 = extractvalue [4 x i32] %b_arr, 0
  %b2 = extractvalue [4 x i32] %b_arr, 2
  %b1 = extractvalue [4 x i32] %b_arr, 1
  %a3 = extractvalue [4 x i32] %a_arr, 3
  %c1 = sub i32 %a1, %b1
  %b3 = extractvalue [4 x i32] %b_arr, 3
  %c0 = sub i32 %a0, %b0
  %c2 = sub i32 %a2, %b2
  %c_arr0 = insertvalue [4 x i32] undef, i32 %c0, 0
  %c_arr1 = insertvalue [4 x i32] %c_arr0, i32 %c1, 1
  %c3 = sub i32 %a3, %b3
  %c_arr2 = insertvalue [4 x i32] %c_arr1, i32 %c2, 2
  %c_arr3 = insertvalue [4 x i32] %c_arr2, i32 %c3, 3
  store [4 x i32] %c_arr3, [4 x i32]* %c, align 4
  ret void
}

; Almost identical to previous test, but for type that should NOT be vectorized.
;
; CHECK-LABEL: julia_load_array_of_i16
; CHECK-NOT: i2>
define void @julia_load_array_of_i16([4 x i16]* %a, [4 x i16]* %b, [4 x i16]* %c) {
top:
  %a_arr = load [4 x i16], [4 x i16]* %a, align 4
  %a0 = extractvalue [4 x i16] %a_arr, 0
  %a2 = extractvalue [4 x i16] %a_arr, 2
  %a1 = extractvalue [4 x i16] %a_arr, 1
  %b_arr = load [4 x i16], [4 x i16]* %b, align 4
  %b0 = extractvalue [4 x i16] %b_arr, 0
  %b2 = extractvalue [4 x i16] %b_arr, 2
  %b1 = extractvalue [4 x i16] %b_arr, 1
  %a3 = extractvalue [4 x i16] %a_arr, 3
  %c1 = sub i16 %a1, %b1
  %b3 = extractvalue [4 x i16] %b_arr, 3
  %c0 = sub i16 %a0, %b0
  %c2 = sub i16 %a2, %b2
  %c_arr0 = insertvalue [4 x i16] undef, i16 %c0, 0
  %c_arr1 = insertvalue [4 x i16] %c_arr0, i16 %c1, 1
  %c3 = sub i16 %a3, %b3
  %c_arr2 = insertvalue [4 x i16] %c_arr1, i16 %c2, 2
  %c_arr3 = insertvalue [4 x i16] %c_arr2, i16 %c3, 3
  store [4 x i16] %c_arr3, [4 x i16]* %c, align 4
  ret void
}

%pseudovec = type { float, float, float, float }

; CHECK-LABEL: julia_load_struct_of_float
; CHECK: load <4 x float>
; CHECK: load <4 x float>
; CHECK: fsub <4 x float>
define void @julia_load_struct_of_float(%pseudovec* %a, %pseudovec* %b, %pseudovec* %c) {
top:
  %a_struct = load %pseudovec, %pseudovec* %a, align 4
  %a0 = extractvalue %pseudovec %a_struct, 0
  %a1 = extractvalue %pseudovec %a_struct, 1
  %b_struct = load %pseudovec, %pseudovec* %b, align 4
  %a2 = extractvalue %pseudovec %a_struct, 2
  %b0 = extractvalue %pseudovec %b_struct, 0
  %a3 = extractvalue %pseudovec %a_struct, 3
  %c0 = fsub float %a0, %b0
  %b1 = extractvalue %pseudovec %b_struct, 1
  %b2 = extractvalue %pseudovec %b_struct, 2
  %c1 = fsub float %a1, %b1
  %c_struct0 = insertvalue %pseudovec undef, float %c0, 0
  %b3 = extractvalue %pseudovec %b_struct, 3
  %c3 = fsub float %a3, %b3
  %c_struct1 = insertvalue %pseudovec %c_struct0, float %c1, 1
  %c2 = fsub float %a2, %b2
  %c_struct2 = insertvalue %pseudovec %c_struct1, float %c2, 2
  %c_struct3 = insertvalue %pseudovec %c_struct2, float %c3, 3
  store %pseudovec %c_struct3, %pseudovec* %c, align 4
  ret void
}
