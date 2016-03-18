; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck %s

declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #0

; FUNC-LABEL: {{^}}ds_bpermute:
; CHECK: ds_bpermute_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define void @ds_bpermute(i32 addrspace(1)* %out, i32 %index, i32 %src) nounwind {
  %bpermute = call i32 @llvm.amdgcn.ds.bpermute(i32 %index, i32 %src) #0
  store i32 %bpermute, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bpermute_no_waitcnt_test:
; CHECK: s_cbranch_scc1
; CHECK: ds_bpermute_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; CHECK-NOT: s_waitcnt
define void @bpermute_no_waitcnt_test(i32 addrspace(1)* %out, i32 %cond) {
entry:

  %tmp = icmp eq i32 %cond, 0
  br i1 %tmp, label %if, label %else

if:                                               ; preds = %entry

  %bpermute = call i32 @llvm.amdgcn.ds.bpermute(i32 0, i32 0) #0
  br label %endif

else:                                             ; preds = %entry
  br label %endif

endif:
  %val = phi i32 [ %bpermute, %if ], [0, %else]      ; preds = %else, %if
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

attributes #0 = { nounwind readnone convergent }
