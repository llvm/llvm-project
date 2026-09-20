; RUN: opt -S -mtriple=amdgpu12.50-amd-amdhsa -passes=slp-vectorizer < %s | FileCheck -check-prefixes=GCN,GFX1250 %s
; RUN: opt -S -mtriple=amdgpu12.51-amd-amdhsa -passes=slp-vectorizer < %s | FileCheck -check-prefixes=GCN,GFX1251 %s

; GCN-LABEL: @fadd_combine
; GFX1250: fadd double
; GFX1250: fadd double
; GFX1251: fadd <2 x double>
define amdgpu_kernel void @fadd_combine(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds double, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load double, ptr addrspace(1) %tmp2, align 8
  %tmp4 = fadd double %tmp3, 1.000000e+00
  store double %tmp4, ptr addrspace(1) %tmp2, align 8
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds double, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load double, ptr addrspace(1) %tmp6, align 8
  %tmp8 = fadd double %tmp7, 1.000000e+00
  store double %tmp8, ptr addrspace(1) %tmp6, align 8
  ret void
}

; GCN-LABEL: @fmul_combine
; GFX1250: fmul double
; GFX1250: fmul double
; GFX1251: fmul <2 x double>
define amdgpu_kernel void @fmul_combine(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds double, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load double, ptr addrspace(1) %tmp2, align 8
  %tmp4 = fmul double %tmp3, 1.000000e+00
  store double %tmp4, ptr addrspace(1) %tmp2, align 8
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds double, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load double, ptr addrspace(1) %tmp6, align 8
  %tmp8 = fmul double %tmp7, 1.000000e+00
  store double %tmp8, ptr addrspace(1) %tmp6, align 8
  ret void
}

; GCN-LABEL: @fma_combine
; GFX1250: call double @llvm.fma.f64
; GFX1250: call double @llvm.fma.f64
; GFX1251: call <2 x double> @llvm.fma.v2f64
define amdgpu_kernel void @fma_combine(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds double, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load double, ptr addrspace(1) %tmp2, align 8
  %tmp4 = tail call double @llvm.fma.f64(double %tmp3, double 1.000000e+00, double 1.000000e+00)
  store double %tmp4, ptr addrspace(1) %tmp2, align 8
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds double, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load double, ptr addrspace(1) %tmp6, align 8
  %tmp8 = tail call double @llvm.fma.f64(double %tmp7, double 1.000000e+00, double 1.000000e+00)
  store double %tmp8, ptr addrspace(1) %tmp6, align 8
  ret void
}

; GCN-LABEL: @fmaxnum_combine
; GFX1250: call double @llvm.maximumnum.f64
; GFX1250: call double @llvm.maximumnum.f64
; GFX1251: call <2 x double> @llvm.maximumnum.v2f64
define amdgpu_kernel void @fmaxnum_combine(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds double, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load double, ptr addrspace(1) %tmp2, align 8
  %tmp4 = tail call double @llvm.maximumnum.f64(double %tmp3, double 1.000000e+00)
  store double %tmp4, ptr addrspace(1) %tmp2, align 8
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds double, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load double, ptr addrspace(1) %tmp6, align 8
  %tmp8 = tail call double @llvm.maximumnum.f64(double %tmp7, double 1.000000e+00)
  store double %tmp8, ptr addrspace(1) %tmp6, align 8
  ret void
}

; GCN-LABEL: @fminnum_combine
; GFX1250: call double @llvm.minimumnum.f64
; GFX1250: call double @llvm.minimumnum.f64
; GFX1251: call <2 x double> @llvm.minimumnum.v2f64
define amdgpu_kernel void @fminnum_combine(ptr addrspace(1) %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds double, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load double, ptr addrspace(1) %tmp2, align 8
  %tmp4 = tail call double @llvm.minimumnum.f64(double %tmp3, double 1.000000e+00)
  store double %tmp4, ptr addrspace(1) %tmp2, align 8
  %tmp5 = add nuw nsw i64 %tmp1, 1
  %tmp6 = getelementptr inbounds double, ptr addrspace(1) %arg, i64 %tmp5
  %tmp7 = load double, ptr addrspace(1) %tmp6, align 8
  %tmp8 = tail call double @llvm.minimumnum.f64(double %tmp7, double 1.000000e+00)
  store double %tmp8, ptr addrspace(1) %tmp6, align 8
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare double @llvm.fma.f64(double, double, double)
declare double @llvm.maximumnum.f64(double, double)
declare double @llvm.minimumnum.f64(double, double)
