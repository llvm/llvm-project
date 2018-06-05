; RUN: llc -march=amdgcn < %s

; Tests for a bug in SelectionDAG::UpdateNodeOperands exposed by VectorLegalizer
; where divergence information is not updated.

declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @spam(double addrspace(1)* noalias %arg) {
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds double, double addrspace(1)* %arg, i64 %tmp1
  %tmp3 = load double, double addrspace(1)* %tmp2, align 8
  %tmp4 = fadd double undef, 0.000000e+00
  %tmp5 = insertelement <2 x double> undef, double %tmp4, i64 0
  %tmp6 = insertelement <2 x double> %tmp5, double %tmp3, i64 1
  %tmp7 = insertelement <2 x double> %tmp6, double 0.000000e+00, i64 1
  %tmp8 = fadd <2 x double> zeroinitializer, undef
  %tmp9 = fadd <2 x double> %tmp7, zeroinitializer
  %tmp10 = extractelement <2 x double> %tmp8, i64 0
  %tmp11 = getelementptr inbounds double, double addrspace(1)* %tmp2, i64 2
  store double %tmp10, double addrspace(1)* %tmp11, align 8
  %tmp12 = getelementptr inbounds double, double addrspace(1)* %tmp2, i64 3
  store double undef, double addrspace(1)* %tmp12, align 8
  %tmp13 = extractelement <2 x double> %tmp9, i64 0
  %tmp14 = getelementptr inbounds double, double addrspace(1)* %tmp2, i64 6
  store double %tmp13, double addrspace(1)* %tmp14, align 8
  %tmp15 = getelementptr inbounds double, double addrspace(1)* %tmp2, i64 7
  store double 0.000000e+00, double addrspace(1)* %tmp15, align 8
  ret void
}
