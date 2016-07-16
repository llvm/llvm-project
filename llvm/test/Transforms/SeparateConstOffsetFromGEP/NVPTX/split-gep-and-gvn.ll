; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=PTX
; RUN: opt < %s -S -separate-const-offset-from-gep -reassociate-geps-verify-no-dead-code -gvn | FileCheck %s --check-prefix=IR

; Verifies the SeparateConstOffsetFromGEP pass.
; The following code computes
; *output = array[x][y] + array[x][y+1] + array[x+1][y] + array[x+1][y+1]
;
; We expect SeparateConstOffsetFromGEP to transform it to
;
; float *base = &a[x][y];
; *output = base[0] + base[1] + base[32] + base[33];
;
; so the backend can emit PTX that uses fewer virtual registers.

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"

@array = internal addrspace(3) constant [32 x [32 x float]] zeroinitializer, align 4

define void @sum_of_array(i32 %x, i32 %y, float* nocapture %output) {
.preheader:
  %0 = sext i32 %y to i64
  %1 = sext i32 %x to i64
  %2 = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]] addrspace(3)* @array, i64 0, i64 %1, i64 %0
  %3 = addrspacecast float addrspace(3)* %2 to float*
  %4 = load float, float* %3, align 4
  %5 = fadd float %4, 0.000000e+00
  %6 = add i32 %y, 1
  %7 = sext i32 %6 to i64
  %8 = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]] addrspace(3)* @array, i64 0, i64 %1, i64 %7
  %9 = addrspacecast float addrspace(3)* %8 to float*
  %10 = load float, float* %9, align 4
  %11 = fadd float %5, %10
  %12 = add i32 %x, 1
  %13 = sext i32 %12 to i64
  %14 = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]] addrspace(3)* @array, i64 0, i64 %13, i64 %0
  %15 = addrspacecast float addrspace(3)* %14 to float*
  %16 = load float, float* %15, align 4
  %17 = fadd float %11, %16
  %18 = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]] addrspace(3)* @array, i64 0, i64 %13, i64 %7
  %19 = addrspacecast float addrspace(3)* %18 to float*
  %20 = load float, float* %19, align 4
  %21 = fadd float %17, %20
  store float %21, float* %output, align 4
  ret void
}
; PTX-LABEL: sum_of_array(
; PTX: ld.shared.f32 {{%f[0-9]+}}, {{\[}}[[BASE_REG:%(rd|r)[0-9]+]]{{\]}}
; PTX: ld.shared.f32 {{%f[0-9]+}}, {{\[}}[[BASE_REG]]+4{{\]}}
; PTX: ld.shared.f32 {{%f[0-9]+}}, {{\[}}[[BASE_REG]]+128{{\]}}
; PTX: ld.shared.f32 {{%f[0-9]+}}, {{\[}}[[BASE_REG]]+132{{\]}}

; IR-LABEL: @sum_of_array(
; TODO: GVN is unable to preserve the "inbounds" keyword on the first GEP. Need
; some infrastructure changes to enable such optimizations.
; IR: [[BASE_PTR:%[a-zA-Z0-9]+]] = getelementptr [32 x [32 x float]], [32 x [32 x float]] addrspace(3)* @array, i64 0, i64 %{{[a-zA-Z0-9]+}}, i64 %{{[a-zA-Z0-9]+}}
; IR: getelementptr inbounds float, float addrspace(3)* [[BASE_PTR]], i64 1
; IR: getelementptr inbounds float, float addrspace(3)* [[BASE_PTR]], i64 32
; IR: getelementptr inbounds float, float addrspace(3)* [[BASE_PTR]], i64 33

; @sum_of_array2 is very similar to @sum_of_array. The only difference is in
; the order of "sext" and "add" when computing the array indices. @sum_of_array
; computes add before sext, e.g., array[sext(x + 1)][sext(y + 1)], while
; @sum_of_array2 computes sext before add,
; e.g., array[sext(x) + 1][sext(y) + 1]. SeparateConstOffsetFromGEP should be
; able to extract constant offsets from both forms.
define void @sum_of_array2(i32 %x, i32 %y, float* nocapture %output) {
.preheader:
  %0 = sext i32 %y to i64
  %1 = sext i32 %x to i64
  %2 = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]] addrspace(3)* @array, i64 0, i64 %1, i64 %0
  %3 = addrspacecast float addrspace(3)* %2 to float*
  %4 = load float, float* %3, align 4
  %5 = fadd float %4, 0.000000e+00
  %6 = add i64 %0, 1
  %7 = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]] addrspace(3)* @array, i64 0, i64 %1, i64 %6
  %8 = addrspacecast float addrspace(3)* %7 to float*
  %9 = load float, float* %8, align 4
  %10 = fadd float %5, %9
  %11 = add i64 %1, 1
  %12 = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]] addrspace(3)* @array, i64 0, i64 %11, i64 %0
  %13 = addrspacecast float addrspace(3)* %12 to float*
  %14 = load float, float* %13, align 4
  %15 = fadd float %10, %14
  %16 = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]] addrspace(3)* @array, i64 0, i64 %11, i64 %6
  %17 = addrspacecast float addrspace(3)* %16 to float*
  %18 = load float, float* %17, align 4
  %19 = fadd float %15, %18
  store float %19, float* %output, align 4
  ret void
}
; PTX-LABEL: sum_of_array2(
; PTX: ld.shared.f32 {{%f[0-9]+}}, {{\[}}[[BASE_REG:%(rd|r)[0-9]+]]{{\]}}
; PTX: ld.shared.f32 {{%f[0-9]+}}, {{\[}}[[BASE_REG]]+4{{\]}}
; PTX: ld.shared.f32 {{%f[0-9]+}}, {{\[}}[[BASE_REG]]+128{{\]}}
; PTX: ld.shared.f32 {{%f[0-9]+}}, {{\[}}[[BASE_REG]]+132{{\]}}

; IR-LABEL: @sum_of_array2(
; IR: [[BASE_PTR:%[a-zA-Z0-9]+]] = getelementptr [32 x [32 x float]], [32 x [32 x float]] addrspace(3)* @array, i64 0, i64 %{{[a-zA-Z0-9]+}}, i64 %{{[a-zA-Z0-9]+}}
; IR: getelementptr inbounds float, float addrspace(3)* [[BASE_PTR]], i64 1
; IR: getelementptr inbounds float, float addrspace(3)* [[BASE_PTR]], i64 32
; IR: getelementptr inbounds float, float addrspace(3)* [[BASE_PTR]], i64 33


; This function loads
;   array[zext(x)][zext(y)]
;   array[zext(x)][zext(y +nuw 1)]
;   array[zext(x +nuw 1)][zext(y)]
;   array[zext(x +nuw 1)][zext(y +nuw 1)].
;
; This function is similar to @sum_of_array, but it
; 1) extends array indices using zext instead of sext;
; 2) annotates the addition with "nuw"; otherwise, zext(x + 1) => zext(x) + 1
;    may be invalid.
define void @sum_of_array3(i32 %x, i32 %y, float* nocapture %output) {
.preheader:
  %0 = zext i32 %y to i64
  %1 = zext i32 %x to i64
  %2 = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]] addrspace(3)* @array, i64 0, i64 %1, i64 %0
  %3 = addrspacecast float addrspace(3)* %2 to float*
  %4 = load float, float* %3, align 4
  %5 = fadd float %4, 0.000000e+00
  %6 = add nuw i32 %y, 1
  %7 = zext i32 %6 to i64
  %8 = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]] addrspace(3)* @array, i64 0, i64 %1, i64 %7
  %9 = addrspacecast float addrspace(3)* %8 to float*
  %10 = load float, float* %9, align 4
  %11 = fadd float %5, %10
  %12 = add nuw i32 %x, 1
  %13 = zext i32 %12 to i64
  %14 = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]] addrspace(3)* @array, i64 0, i64 %13, i64 %0
  %15 = addrspacecast float addrspace(3)* %14 to float*
  %16 = load float, float* %15, align 4
  %17 = fadd float %11, %16
  %18 = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]] addrspace(3)* @array, i64 0, i64 %13, i64 %7
  %19 = addrspacecast float addrspace(3)* %18 to float*
  %20 = load float, float* %19, align 4
  %21 = fadd float %17, %20
  store float %21, float* %output, align 4
  ret void
}
; PTX-LABEL: sum_of_array3(
; PTX: ld.shared.f32 {{%f[0-9]+}}, {{\[}}[[BASE_REG:%(rd|r)[0-9]+]]{{\]}}
; PTX: ld.shared.f32 {{%f[0-9]+}}, {{\[}}[[BASE_REG]]+4{{\]}}
; PTX: ld.shared.f32 {{%f[0-9]+}}, {{\[}}[[BASE_REG]]+128{{\]}}
; PTX: ld.shared.f32 {{%f[0-9]+}}, {{\[}}[[BASE_REG]]+132{{\]}}

; IR-LABEL: @sum_of_array3(
; IR: [[BASE_PTR:%[a-zA-Z0-9]+]] = getelementptr [32 x [32 x float]], [32 x [32 x float]] addrspace(3)* @array, i64 0, i64 %{{[a-zA-Z0-9]+}}, i64 %{{[a-zA-Z0-9]+}}
; IR: getelementptr inbounds float, float addrspace(3)* [[BASE_PTR]], i64 1
; IR: getelementptr inbounds float, float addrspace(3)* [[BASE_PTR]], i64 32
; IR: getelementptr inbounds float, float addrspace(3)* [[BASE_PTR]], i64 33


; This function loads
;   array[zext(x)][zext(y)]
;   array[zext(x)][zext(y)]
;   array[zext(x) + 1][zext(y) + 1]
;   array[zext(x) + 1][zext(y) + 1].
;
; We expect the generated code to reuse the computation of
; &array[zext(x)][zext(y)]. See the expected IR and PTX for details.
define void @sum_of_array4(i32 %x, i32 %y, float* nocapture %output) {
.preheader:
  %0 = zext i32 %y to i64
  %1 = zext i32 %x to i64
  %2 = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]] addrspace(3)* @array, i64 0, i64 %1, i64 %0
  %3 = addrspacecast float addrspace(3)* %2 to float*
  %4 = load float, float* %3, align 4
  %5 = fadd float %4, 0.000000e+00
  %6 = add i64 %0, 1
  %7 = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]] addrspace(3)* @array, i64 0, i64 %1, i64 %6
  %8 = addrspacecast float addrspace(3)* %7 to float*
  %9 = load float, float* %8, align 4
  %10 = fadd float %5, %9
  %11 = add i64 %1, 1
  %12 = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]] addrspace(3)* @array, i64 0, i64 %11, i64 %0
  %13 = addrspacecast float addrspace(3)* %12 to float*
  %14 = load float, float* %13, align 4
  %15 = fadd float %10, %14
  %16 = getelementptr inbounds [32 x [32 x float]], [32 x [32 x float]] addrspace(3)* @array, i64 0, i64 %11, i64 %6
  %17 = addrspacecast float addrspace(3)* %16 to float*
  %18 = load float, float* %17, align 4
  %19 = fadd float %15, %18
  store float %19, float* %output, align 4
  ret void
}
; PTX-LABEL: sum_of_array4(
; PTX: ld.shared.f32 {{%f[0-9]+}}, {{\[}}[[BASE_REG:%(rd|r)[0-9]+]]{{\]}}
; PTX: ld.shared.f32 {{%f[0-9]+}}, {{\[}}[[BASE_REG]]+4{{\]}}
; PTX: ld.shared.f32 {{%f[0-9]+}}, {{\[}}[[BASE_REG]]+128{{\]}}
; PTX: ld.shared.f32 {{%f[0-9]+}}, {{\[}}[[BASE_REG]]+132{{\]}}

; IR-LABEL: @sum_of_array4(
; IR: [[BASE_PTR:%[a-zA-Z0-9]+]] = getelementptr [32 x [32 x float]], [32 x [32 x float]] addrspace(3)* @array, i64 0, i64 %{{[a-zA-Z0-9]+}}, i64 %{{[a-zA-Z0-9]+}}
; IR: getelementptr inbounds float, float addrspace(3)* [[BASE_PTR]], i64 1
; IR: getelementptr inbounds float, float addrspace(3)* [[BASE_PTR]], i64 32
; IR: getelementptr inbounds float, float addrspace(3)* [[BASE_PTR]], i64 33


; The source code is:
;   p0 = &input[sext(x + y)];
;   p1 = &input[sext(x + (y + 5))];
;
; Without reuniting extensions, SeparateConstOffsetFromGEP would emit
;   p0 = &input[sext(x + y)];
;   t1 = &input[sext(x) + sext(y)];
;   p1 = &t1[5];
;
; With reuniting extensions, it merges p0 and t1 and thus emits
;   p0 = &input[sext(x + y)];
;   p1 = &p0[5];
define void @reunion(i32 %x, i32 %y, float* %input) {
; IR-LABEL: @reunion(
; PTX-LABEL: reunion(
entry:
  %xy = add nsw i32 %x, %y
  %0 = sext i32 %xy to i64
  %p0 = getelementptr inbounds float, float* %input, i64 %0
  %v0 = load float, float* %p0, align 4
; PTX: ld.f32 %f{{[0-9]+}}, {{\[}}[[p0:%rd[0-9]+]]{{\]}}
  call void @use(float %v0)

  %y5 = add nsw i32 %y, 5
  %xy5 = add nsw i32 %x, %y5
  %1 = sext i32 %xy5 to i64
  %p1 = getelementptr inbounds float, float* %input, i64 %1
; IR: getelementptr inbounds float, float* %p0, i64 5
  %v1 = load float, float* %p1, align 4
; PTX: ld.f32 %f{{[0-9]+}}, {{\[}}[[p0]]+20{{\]}}
  call void @use(float %v1)

  ret void
}

declare void @use(float)
