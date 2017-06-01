; RUN: llc -march=amdgcn < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

; FUNC-LABEL: @v_test_imax3_sgt_i32
; SI: v_max3_i32
define void @v_test_imax3_sgt_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr, i32 addrspace(1)* %cptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr i32, i32 addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0, align 4
  %b = load i32, i32 addrspace(1)* %gep1, align 4
  %c = load i32, i32 addrspace(1)* %gep2, align 4
  %icmp0 = icmp sgt i32 %a, %b
  %i0 = select i1 %icmp0, i32 %a, i32 %b
  %icmp1 = icmp sgt i32 %i0, %c
  %i1 = select i1 %icmp1, i32 %i0, i32 %c
  store i32 %i1, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_test_umax3_ugt_i32
; SI: v_max3_u32
define void @v_test_umax3_ugt_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr, i32 addrspace(1)* %cptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr i32, i32 addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0, align 4
  %b = load i32, i32 addrspace(1)* %gep1, align 4
  %c = load i32, i32 addrspace(1)* %gep2, align 4
  %icmp0 = icmp ugt i32 %a, %b
  %i0 = select i1 %icmp0, i32 %a, i32 %b
  %icmp1 = icmp ugt i32 %i0, %c
  %i1 = select i1 %icmp1, i32 %i0, i32 %c
  store i32 %i1, i32 addrspace(1)* %out, align 4
  ret void
}
