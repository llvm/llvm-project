; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -infer-address-spaces %s | FileCheck %s

; Test that pure GetElementPtr instructions not directly connected to
; a memory operation are inferred.

@lds = internal unnamed_addr addrspace(3) global [648 x double] undef, align 8

; CHECK-LABEL: @simplified_constexpr_gep_addrspacecast(
; CHECK: %gep0 = getelementptr inbounds double, double addrspace(3)* getelementptr inbounds ([648 x double], [648 x double] addrspace(3)* @lds, i64 0, i64 384), i64 %idx0
; CHECK-NEXT: store double 1.000000e+00, double addrspace(3)* %gep0, align 8
define void @simplified_constexpr_gep_addrspacecast(i64 %idx0, i64 %idx1) {
  %gep0 = getelementptr inbounds double, double addrspace(4)* addrspacecast (double addrspace(3)* getelementptr inbounds ([648 x double], [648 x double] addrspace(3)* @lds, i64 0, i64 384) to double addrspace(4)*), i64 %idx0
  %asc = addrspacecast double addrspace(4)* %gep0 to double addrspace(3)*
  store double 1.000000e+00, double addrspace(3)* %asc, align 8
  ret void
}

; CHECK-LABEL: @constexpr_gep_addrspacecast(
; CHECK-NEXT: %gep0 = getelementptr inbounds double, double addrspace(3)* getelementptr inbounds ([648 x double], [648 x double] addrspace(3)* @lds, i64 0, i64 384), i64 %idx0
; CHECK-NEXT: store double 1.000000e+00, double addrspace(3)* %gep0, align 8
define void @constexpr_gep_addrspacecast(i64 %idx0, i64 %idx1) {
  %gep0 = getelementptr inbounds double, double addrspace(4)* getelementptr ([648 x double], [648 x double] addrspace(4)* addrspacecast ([648 x double] addrspace(3)* @lds to [648 x double] addrspace(4)*), i64 0, i64 384), i64 %idx0
  %asc = addrspacecast double addrspace(4)* %gep0 to double addrspace(3)*
  store double 1.0, double addrspace(3)* %asc, align 8
  ret void
}

; CHECK-LABEL: @constexpr_gep_gep_addrspacecast(
; CHECK: %gep0 = getelementptr inbounds double, double addrspace(3)* getelementptr inbounds ([648 x double], [648 x double] addrspace(3)* @lds, i64 0, i64 384), i64 %idx0
; CHECK-NEXT: %gep1 = getelementptr inbounds double, double addrspace(3)* %gep0, i64 %idx1
; CHECK-NEXT: store double 1.000000e+00, double addrspace(3)* %gep1, align 8
define void @constexpr_gep_gep_addrspacecast(i64 %idx0, i64 %idx1) {
  %gep0 = getelementptr inbounds double, double addrspace(4)* getelementptr ([648 x double], [648 x double] addrspace(4)* addrspacecast ([648 x double] addrspace(3)* @lds to [648 x double] addrspace(4)*), i64 0, i64 384), i64 %idx0
  %gep1 = getelementptr inbounds double, double addrspace(4)* %gep0, i64 %idx1
  %asc = addrspacecast double addrspace(4)* %gep1 to double addrspace(3)*
  store double 1.0, double addrspace(3)* %asc, align 8
  ret void
}

; Don't crash
; CHECK-LABEL: @vector_gep(
; CHECK: %cast = addrspacecast <4 x [1024 x i32] addrspace(3)*> %array to <4 x [1024 x i32] addrspace(4)*>
define amdgpu_kernel void @vector_gep(<4 x [1024 x i32] addrspace(3)*> %array) nounwind {
  %cast = addrspacecast <4 x [1024 x i32] addrspace(3)*> %array to <4 x [1024 x i32] addrspace(4)*>
  %p = getelementptr [1024 x i32], <4 x [1024 x i32] addrspace(4)*> %cast, <4 x i16> zeroinitializer, <4 x i16> <i16 16, i16 16, i16 16, i16 16>
  %p0 = extractelement <4 x i32 addrspace(4)*> %p, i32 0
  %p1 = extractelement <4 x i32 addrspace(4)*> %p, i32 1
  %p2 = extractelement <4 x i32 addrspace(4)*> %p, i32 2
  %p3 = extractelement <4 x i32 addrspace(4)*> %p, i32 3
  store i32 99, i32 addrspace(4)* %p0
  store i32 99, i32 addrspace(4)* %p1
  store i32 99, i32 addrspace(4)* %p2
  store i32 99, i32 addrspace(4)* %p3
  ret void
}

; CHECK-LABEL: @repeated_constexpr_gep_addrspacecast(
; CHECK-NEXT: %gep0 = getelementptr inbounds double, double addrspace(3)* getelementptr inbounds ([648 x double], [648 x double] addrspace(3)* @lds, i64 0, i64 384), i64 %idx0
; CHECK-NEXT: store double 1.000000e+00, double addrspace(3)* %gep0, align 8
; CHECK-NEXT: %gep1 = getelementptr inbounds double, double addrspace(3)* getelementptr inbounds ([648 x double], [648 x double] addrspace(3)* @lds, i64 0, i64 384), i64 %idx1
; CHECK-NEXT: store double 1.000000e+00, double addrspace(3)* %gep1, align 8
; CHECK-NEXT: ret void
define void @repeated_constexpr_gep_addrspacecast(i64 %idx0, i64 %idx1) {
  %gep0 = getelementptr inbounds double, double addrspace(4)* getelementptr ([648 x double], [648 x double] addrspace(4)* addrspacecast ([648 x double] addrspace(3)* @lds to [648 x double] addrspace(4)*), i64 0, i64 384), i64 %idx0
  %asc0 = addrspacecast double addrspace(4)* %gep0 to double addrspace(3)*
  store double 1.0, double addrspace(3)* %asc0, align 8

  %gep1 = getelementptr inbounds double, double addrspace(4)* getelementptr ([648 x double], [648 x double] addrspace(4)* addrspacecast ([648 x double] addrspace(3)* @lds to [648 x double] addrspace(4)*), i64 0, i64 384), i64 %idx1
  %asc1 = addrspacecast double addrspace(4)* %gep1 to double addrspace(3)*
  store double 1.0, double addrspace(3)* %asc1, align 8

  ret void
}
