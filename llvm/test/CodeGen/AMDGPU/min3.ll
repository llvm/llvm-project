; RUN: llc -march=amdgcn < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

; FUNC-LABEL: @v_test_imin3_slt_i32
; SI: v_min3_i32
define void @v_test_imin3_slt_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr, i32 addrspace(1)* %cptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr i32, i32 addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0, align 4
  %b = load i32, i32 addrspace(1)* %gep1, align 4
  %c = load i32, i32 addrspace(1)* %gep2, align 4
  %icmp0 = icmp slt i32 %a, %b
  %i0 = select i1 %icmp0, i32 %a, i32 %b
  %icmp1 = icmp slt i32 %i0, %c
  %i1 = select i1 %icmp1, i32 %i0, i32 %c
  store i32 %i1, i32 addrspace(1)* %outgep, align 4
  ret void
}

; FUNC-LABEL: @v_test_umin3_ult_i32
; SI: v_min3_u32
define void @v_test_umin3_ult_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr, i32 addrspace(1)* %cptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr i32, i32 addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0, align 4
  %b = load i32, i32 addrspace(1)* %gep1, align 4
  %c = load i32, i32 addrspace(1)* %gep2, align 4
  %icmp0 = icmp ult i32 %a, %b
  %i0 = select i1 %icmp0, i32 %a, i32 %b
  %icmp1 = icmp ult i32 %i0, %c
  %i1 = select i1 %icmp1, i32 %i0, i32 %c
  store i32 %i1, i32 addrspace(1)* %outgep, align 4
  ret void
}

; FUNC-LABEL: @v_test_umin_umin_umin
; SI: v_min_i32
; SI: v_min3_i32
define void @v_test_umin_umin_umin(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr, i32 addrspace(1)* %cptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %tid2 = mul i32 %tid, 2
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr i32, i32 addrspace(1)* %cptr, i32 %tid

  %gep3 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid2
  %gep4 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid2
  %gep5 = getelementptr i32, i32 addrspace(1)* %cptr, i32 %tid2

  %outgep0 = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %outgep1 = getelementptr i32, i32 addrspace(1)* %out, i32 %tid2

  %a = load i32, i32 addrspace(1)* %gep0, align 4
  %b = load i32, i32 addrspace(1)* %gep1, align 4
  %c = load i32, i32 addrspace(1)* %gep2, align 4
  %d = load i32, i32 addrspace(1)* %gep3, align 4

  %icmp0 = icmp slt i32 %a, %b
  %i0 = select i1 %icmp0, i32 %a, i32 %b

  %icmp1 = icmp slt i32 %c, %d
  %i1 = select i1 %icmp1, i32 %c, i32 %d

  %icmp2 = icmp slt i32 %i0, %i1
  %i2 = select i1 %icmp2, i32 %i0, i32 %i1

  store i32 %i2, i32 addrspace(1)* %outgep1, align 4
  ret void
}

; FUNC-LABEL: @v_test_umin3_2_uses
; SI-NOT: v_min3
define void @v_test_umin3_2_uses(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr, i32 addrspace(1)* %cptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %tid2 = mul i32 %tid, 2
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr i32, i32 addrspace(1)* %cptr, i32 %tid

  %gep3 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid2
  %gep4 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid2
  %gep5 = getelementptr i32, i32 addrspace(1)* %cptr, i32 %tid2

  %outgep0 = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %outgep1 = getelementptr i32, i32 addrspace(1)* %out, i32 %tid2

  %a = load i32, i32 addrspace(1)* %gep0, align 4
  %b = load i32, i32 addrspace(1)* %gep1, align 4
  %c = load i32, i32 addrspace(1)* %gep2, align 4
  %d = load i32, i32 addrspace(1)* %gep3, align 4

  %icmp0 = icmp slt i32 %a, %b
  %i0 = select i1 %icmp0, i32 %a, i32 %b

  %icmp1 = icmp slt i32 %c, %d
  %i1 = select i1 %icmp1, i32 %c, i32 %d

  %icmp2 = icmp slt i32 %i0, %c
  %i2 = select i1 %icmp2, i32 %i0, i32 %c

  store i32 %i2, i32 addrspace(1)* %outgep0, align 4
  store i32 %i0, i32 addrspace(1)* %outgep1, align 4
  ret void
}
