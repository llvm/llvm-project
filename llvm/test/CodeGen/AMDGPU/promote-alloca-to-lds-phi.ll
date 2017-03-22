; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -amdgpu-promote-alloca < %s | FileCheck %s


; CHECK-LABEL: @branch_ptr_var_same_alloca(
; CHECK: getelementptr inbounds [256 x [64 x i32]], [256 x [64 x i32]] addrspace(3)* @branch_ptr_var_same_alloca.alloca, i32 0, i32 %{{[0-9]+}}

; CHECK: if:
; CHECK: %arrayidx0 = getelementptr inbounds [64 x i32], [64 x i32] addrspace(3)* %{{[0-9]+}}, i32 0, i32 %a

; CHECK: else:
; CHECK: %arrayidx1 = getelementptr inbounds [64 x i32], [64 x i32] addrspace(3)* %15, i32 0, i32 %b

; CHECK: endif:
; CHECK: %phi.ptr = phi i32 addrspace(3)* [ %arrayidx0, %if ], [ %arrayidx1, %else ]
; CHECK: store i32 0, i32 addrspace(3)* %phi.ptr, align 4
define amdgpu_kernel void @branch_ptr_var_same_alloca(i32 %a, i32 %b) #0 {
entry:
  %alloca = alloca [64 x i32], align 4
  br i1 undef, label %if, label %else

if:
  %arrayidx0 = getelementptr inbounds [64 x i32], [64 x i32]* %alloca, i32 0, i32 %a
  br label %endif

else:
  %arrayidx1 = getelementptr inbounds [64 x i32], [64 x i32]* %alloca, i32 0, i32 %b
  br label %endif

endif:
  %phi.ptr = phi i32* [ %arrayidx0, %if ], [ %arrayidx1, %else ]
  store i32 0, i32* %phi.ptr, align 4
  ret void
}

; CHECK-LABEL: @branch_ptr_phi_alloca_null_0(
; CHECK: %phi.ptr = phi i32 addrspace(3)* [ %arrayidx0, %if ], [ null, %entry ]
define amdgpu_kernel void @branch_ptr_phi_alloca_null_0(i32 %a, i32 %b) #0 {
entry:
  %alloca = alloca [64 x i32], align 4
  br i1 undef, label %if, label %endif

if:
  %arrayidx0 = getelementptr inbounds [64 x i32], [64 x i32]* %alloca, i32 0, i32 %a
  br label %endif

endif:
  %phi.ptr = phi i32* [ %arrayidx0, %if ], [ null, %entry ]
  store i32 0, i32* %phi.ptr, align 4
  ret void
}

; CHECK-LABEL: @branch_ptr_phi_alloca_null_1(
; CHECK: %phi.ptr = phi i32 addrspace(3)*  [ null, %entry ], [ %arrayidx0, %if ]
define amdgpu_kernel void @branch_ptr_phi_alloca_null_1(i32 %a, i32 %b) #0 {
entry:
  %alloca = alloca [64 x i32], align 4
  br i1 undef, label %if, label %endif

if:
  %arrayidx0 = getelementptr inbounds [64 x i32], [64 x i32]* %alloca, i32 0, i32 %a
  br label %endif

endif:
  %phi.ptr = phi i32* [ null, %entry ], [ %arrayidx0, %if ]
  store i32 0, i32* %phi.ptr, align 4
  ret void
}

; CHECK-LABEL: @one_phi_value(
; CHECK: getelementptr inbounds [256 x [64 x i32]], [256 x [64 x i32]] addrspace(3)* @one_phi_value.alloca, i32 0, i32 %14
; CHECK:  %arrayidx0 = getelementptr inbounds [64 x i32], [64 x i32] addrspace(3)* %{{[0-9]+}}, i32 0, i32 %a

; CHECK: br label %exit
; CHECK: %phi.ptr = phi i32 addrspace(3)* [ %arrayidx0, %entry ]
; CHECK: store i32 0, i32 addrspace(3)* %phi.ptr, align 4
define amdgpu_kernel void @one_phi_value(i32 %a) #0 {
entry:
  %alloca = alloca [64 x i32], align 4
  %arrayidx0 = getelementptr inbounds [64 x i32], [64 x i32]* %alloca, i32 0, i32 %a
  br label %exit

exit:
  %phi.ptr = phi i32* [ %arrayidx0, %entry ]
  store i32 0, i32* %phi.ptr, align 4
  ret void
}

; CHECK-LABEL: @branch_ptr_alloca_unknown_obj(
; CHECK: %alloca = alloca [64 x i32], align 4

; CHECK: if:
; CHECK: %arrayidx0 = getelementptr inbounds [64 x i32], [64 x i32]* %alloca, i32 0, i32 %a

; CHECK: else:
; CHECK: %arrayidx1 = call i32* @get_unknown_pointer()

; CHECK: endif:
; CHECK: %phi.ptr = phi i32* [ %arrayidx0, %if ], [ %arrayidx1, %else ]
; CHECK: store i32 0, i32* %phi.ptr, align 4
define amdgpu_kernel void @branch_ptr_alloca_unknown_obj(i32 %a, i32 %b) #0 {
entry:
  %alloca = alloca [64 x i32], align 4
  br i1 undef, label %if, label %else

if:
  %arrayidx0 = getelementptr inbounds [64 x i32], [64 x i32]* %alloca, i32 0, i32 %a
  br label %endif

else:
  %arrayidx1 = call i32* @get_unknown_pointer()
  br label %endif

endif:
  %phi.ptr = phi i32* [ %arrayidx0, %if ], [ %arrayidx1, %else ]
  store i32 0, i32* %phi.ptr, align 4
  ret void
}

; kernel void ptr_induction_var_same_alloca(void)
; {
;     int alloca[64];
;     int i = 0;

;     #pragma nounroll
;     for (int* p = &alloca[2], *e = &alloca[48]; p != e; ++p, ++i)
;     {
;         *p = i;
;     }
; }

; FIXME: This should be promotable. We need to use
; GetUnderlyingObjects when looking at the icmp user.

; CHECK-LABEL: @ptr_induction_var_same_alloca(
; CHECK: %alloca = alloca [64 x i32], align 4
; CHECK: phi i32* [ %arrayidx, %entry ], [ %incdec.ptr, %for.body ]
define amdgpu_kernel void @ptr_induction_var_same_alloca() #0 {
entry:
  %alloca = alloca [64 x i32], align 4
  %arrayidx = getelementptr inbounds [64 x i32], [64 x i32]* %alloca, i32 0, i32 2
  %arrayidx1 = getelementptr inbounds [64 x i32], [64 x i32]* %alloca, i32 0, i32 48
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %i.09 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %p.08 = phi i32* [ %arrayidx, %entry ], [ %incdec.ptr, %for.body ]
  store i32 %i.09, i32* %p.08, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %p.08, i32 1
  %inc = add nuw nsw i32 %i.09, 1
  %cmp = icmp eq i32* %incdec.ptr, %arrayidx1
  br i1 %cmp, label %for.cond.cleanup, label %for.body
}


; extern int* get_unknown_pointer(void);

; kernel void ptr_induction_var_alloca_unknown(void)
; {
;     int alloca[64];
;     int i = 0;
;
;     for (int* p = &alloca[2], *e = get_unknown_pointer(); p != e; ++p, ++i)
;     {
;         *p = i;
;     }
; }

; CHECK-LABEL: @ptr_induction_var_alloca_unknown(
; CHECK: %alloca = alloca [64 x i32], align 4
; CHECK: %p.08 = phi i32* [ %incdec.ptr, %for.body ], [ %arrayidx, %for.body.preheader ]
; CHECK: %cmp = icmp eq i32* %incdec.ptr, %call
define amdgpu_kernel void @ptr_induction_var_alloca_unknown() #0 {
entry:
  %alloca = alloca [64 x i32], align 4
  %arrayidx = getelementptr inbounds [64 x i32], [64 x i32]* %alloca, i32 0, i32 2
  %call = tail call i32* @get_unknown_pointer() #2
  %cmp.7 = icmp eq i32* %arrayidx, %call
  br i1 %cmp.7, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.preheader
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %p.08 = phi i32* [ %incdec.ptr, %for.body ], [ %arrayidx, %for.body.preheader ]
  store i32 %i.09, i32* %p.08, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %p.08, i32 1
  %inc = add nuw nsw i32 %i.09, 1
  %cmp = icmp eq i32* %incdec.ptr, %call
  br i1 %cmp, label %for.cond.cleanup.loopexit, label %for.body
}

declare i32* @get_unknown_pointer() #0

attributes #0 = { nounwind "amdgpu-waves-per-eu"="1,1" }
