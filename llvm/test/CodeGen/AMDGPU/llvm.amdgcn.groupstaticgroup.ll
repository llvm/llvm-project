; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck %s


@lds0 = addrspace(3) global [512 x float] undef, align 4
@lds1 = addrspace(3) global [256 x float] undef, align 4

; FUNC-LABEL: {{^}}groupstaticsize_test0:
; CHECK: s_movk_i32 s{{[0-9]+}}, 0x800
define void @get_groupstaticsize_test0(float addrspace(1)* %out, i32 addrspace(1)* %lds_size) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 64
  %static_lds_size = call i32 @llvm.amdgcn.groupstaticsize() #1
  store i32 %static_lds_size, i32 addrspace(1)* %lds_size, align 4
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds0, i32 0, i32 %idx.0
  %val0 = load float, float addrspace(3)* %arrayidx0, align 4
  store float %val0, float addrspace(1)* %out, align 4

  ret void
}


; FUNC-LABEL: {{^}}groupstaticsize_test1:
; CHECK: s_movk_i32 s{{[0-9]+}}, 0xc00
define void @groupstaticsize_test1(float addrspace(1)* %out, i32 %cond, i32 addrspace(1)* %lds_size) {
entry:
  %static_lds_size = call i32 @llvm.amdgcn.groupstaticsize() #1
  store i32 %static_lds_size, i32 addrspace(1)* %lds_size, align 4
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 64
  %tmp = icmp eq i32 %cond, 0
  br i1 %tmp, label %if, label %else

if:                                               ; preds = %entry
  %arrayidx0 = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds0, i32 0, i32 %idx.0
  %val0 = load float, float addrspace(3)* %arrayidx0, align 4
  store float %val0, float addrspace(1)* %out, align 4
  br label %endif

else:                                             ; preds = %entry
  %arrayidx1 = getelementptr inbounds [256 x float], [256 x float] addrspace(3)* @lds1, i32 0, i32 %idx.0
  %val1 = load float, float addrspace(3)* %arrayidx1, align 4
  store float %val1, float addrspace(1)* %out, align 4
  br label %endif

endif:                                            ; preds = %else, %if
  ret void
}


declare i32 @llvm.amdgcn.groupstaticsize() #1
declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
