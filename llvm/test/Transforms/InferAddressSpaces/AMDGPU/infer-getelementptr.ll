; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=infer-address-spaces %s | FileCheck %s

; Test that pure GetElementPtr instructions not directly connected to
; a memory operation are inferred.

@lds = internal unnamed_addr addrspace(3) global [648 x double] undef, align 8

; CHECK-LABEL: @simplified_constexpr_gep_addrspacecast(
; CHECK: %gep0 = getelementptr inbounds double, ptr addrspace(3) getelementptr inbounds ([648 x double], ptr addrspace(3) @lds, i64 0, i64 384), i64 %idx0
; CHECK-NEXT: store double 1.000000e+00, ptr addrspace(3) %gep0, align 8
define void @simplified_constexpr_gep_addrspacecast(i64 %idx0, i64 %idx1) {
  %gep0 = getelementptr inbounds double, ptr addrspacecast (ptr addrspace(3) getelementptr inbounds ([648 x double], ptr addrspace(3) @lds, i64 0, i64 384) to ptr), i64 %idx0
  %asc = addrspacecast ptr %gep0 to ptr addrspace(3)
  store double 1.000000e+00, ptr addrspace(3) %asc, align 8
  ret void
}

; CHECK-LABEL: @constexpr_gep_addrspacecast(
; CHECK-NEXT: %gep0 = getelementptr inbounds double, ptr addrspace(3) getelementptr inbounds ([648 x double], ptr addrspace(3) @lds, i64 0, i64 384), i64 %idx0
; CHECK-NEXT: store double 1.000000e+00, ptr addrspace(3) %gep0, align 8
define void @constexpr_gep_addrspacecast(i64 %idx0, i64 %idx1) {
  %gep0 = getelementptr inbounds double, ptr getelementptr ([648 x double], ptr addrspacecast (ptr addrspace(3) @lds to ptr), i64 0, i64 384), i64 %idx0
  %asc = addrspacecast ptr %gep0 to ptr addrspace(3)
  store double 1.0, ptr addrspace(3) %asc, align 8
  ret void
}

; CHECK-LABEL: @constexpr_gep_gep_addrspacecast(
; CHECK: %gep0 = getelementptr inbounds double, ptr addrspace(3) getelementptr inbounds ([648 x double], ptr addrspace(3) @lds, i64 0, i64 384), i64 %idx0
; CHECK-NEXT: %gep1 = getelementptr inbounds double, ptr addrspace(3) %gep0, i64 %idx1
; CHECK-NEXT: store double 1.000000e+00, ptr addrspace(3) %gep1, align 8
define void @constexpr_gep_gep_addrspacecast(i64 %idx0, i64 %idx1) {
  %gep0 = getelementptr inbounds double, ptr getelementptr ([648 x double], ptr addrspacecast (ptr addrspace(3) @lds to ptr), i64 0, i64 384), i64 %idx0
  %gep1 = getelementptr inbounds double, ptr %gep0, i64 %idx1
  %asc = addrspacecast ptr %gep1 to ptr addrspace(3)
  store double 1.0, ptr addrspace(3) %asc, align 8
  ret void
}

; Don't crash
; CHECK-LABEL: @vector_gep(
; CHECK: %cast = addrspacecast <4 x ptr addrspace(3)> %array to <4 x ptr>
define amdgpu_kernel void @vector_gep(<4 x ptr addrspace(3)> %array) nounwind {
  %cast = addrspacecast <4 x ptr addrspace(3)> %array to <4 x ptr>
  %p = getelementptr [1024 x i32], <4 x ptr> %cast, <4 x i16> zeroinitializer, <4 x i16> <i16 16, i16 16, i16 16, i16 16>
  %p0 = extractelement <4 x ptr> %p, i32 0
  %p1 = extractelement <4 x ptr> %p, i32 1
  %p2 = extractelement <4 x ptr> %p, i32 2
  %p3 = extractelement <4 x ptr> %p, i32 3
  store i32 99, ptr %p0
  store i32 99, ptr %p1
  store i32 99, ptr %p2
  store i32 99, ptr %p3
  ret void
}

; CHECK-LABEL: @repeated_constexpr_gep_addrspacecast(
; CHECK-NEXT: %gep0 = getelementptr inbounds double, ptr addrspace(3) getelementptr inbounds ([648 x double], ptr addrspace(3) @lds, i64 0, i64 384), i64 %idx0
; CHECK-NEXT: store double 1.000000e+00, ptr addrspace(3) %gep0, align 8
; CHECK-NEXT: %gep1 = getelementptr inbounds double, ptr addrspace(3) getelementptr inbounds ([648 x double], ptr addrspace(3) @lds, i64 0, i64 384), i64 %idx1
; CHECK-NEXT: store double 1.000000e+00, ptr addrspace(3) %gep1, align 8
; CHECK-NEXT: ret void
define void @repeated_constexpr_gep_addrspacecast(i64 %idx0, i64 %idx1) {
  %gep0 = getelementptr inbounds double, ptr getelementptr ([648 x double], ptr addrspacecast (ptr addrspace(3) @lds to ptr), i64 0, i64 384), i64 %idx0
  %asc0 = addrspacecast ptr %gep0 to ptr addrspace(3)
  store double 1.0, ptr addrspace(3) %asc0, align 8

  %gep1 = getelementptr inbounds double, ptr getelementptr ([648 x double], ptr addrspacecast (ptr addrspace(3) @lds to ptr), i64 0, i64 384), i64 %idx1
  %asc1 = addrspacecast ptr %gep1 to ptr addrspace(3)
  store double 1.0, ptr addrspace(3) %asc1, align 8

  ret void
}

; CHECK-LABEL: @unorder_constexpr_gep_bitcast(
; CHECK-NEXT: %x0 = load i32, ptr addrspace(3) @lds, align 4
; CHECK-NEXT: %x1 = load i32, ptr addrspace(3) getelementptr inbounds (i32, ptr addrspace(3) @lds, i32 1), align 4
define void @unorder_constexpr_gep_bitcast() {
  %x0 = load i32, ptr addrspacecast (ptr addrspace(3) @lds to ptr), align 4
  %x1 = load i32, ptr getelementptr (i32, ptr addrspacecast (ptr addrspace(3) @lds to ptr), i32 1), align 4
  call void @use(i32 %x0, i32 %x1)
  ret void
}

declare void @use(i32, i32)
