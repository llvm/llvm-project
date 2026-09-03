; RUN: split-file %s %t
; RUN: not llc -mtriple=amdgpu12.50 -filetype=null %t/oob.ll 2>&1 | FileCheck %s -check-prefix=OOB
; RUN: not llc -mtriple=amdgpu12.50 -filetype=null %t/oob-wait.ll 2>&1 | FileCheck %s -check-prefix=OOB-WAIT
; RUN: not llc -mtriple=amdgpu12.50 -filetype=null %t/reserved.ll 2>&1 | FileCheck %s -check-prefix=RESERVED
; RUN: not llc -mtriple=amdgpu12.50 -filetype=null %t/reserved-wait.ll 2>&1 | FileCheck %s -check-prefix=RESERVED-WAIT

; The stage operand carries a RangeSet covering the valid, non-reserved stages,
; so the IR verifier rejects bad stages before instruction selection. Reserved
; stages are simply the gaps in that set.

;--- oob.ll
; Values above ALL, and values between STAGE_LAST and ALL, are out of range.
; OOB: immarg value 11 for arg 0 out of range set
; OOB: immarg value 17 for arg 0 out of range set
define amdgpu_kernel void @asyncmark_out_of_range() {
  call void @llvm.amdgcn.asyncmark(i32 11)
  call void @llvm.amdgcn.asyncmark(i32 17)
  ret void
}

;--- oob-wait.ll
; OOB-WAIT: immarg value 15 for arg 1 out of range set
define amdgpu_kernel void @wait_asyncmark_out_of_range() {
  call void @llvm.amdgcn.wait.asyncmark(i16 0, i32 15)
  ret void
}

;--- reserved.ll
; Every reserved stage is rejected. These hold slots in the stage numbering for
; async operations that do not exist yet.
; RESERVED: immarg value 4 for arg 0 out of range set
; RESERVED: immarg value 6 for arg 0 out of range set
; RESERVED: immarg value 7 for arg 0 out of range set
; RESERVED: immarg value 8 for arg 0 out of range set
; RESERVED: immarg value 9 for arg 0 out of range set
; RESERVED: immarg value 10 for arg 0 out of range set
define amdgpu_kernel void @asyncmark_reserved() {
  call void @llvm.amdgcn.asyncmark(i32 4)
  call void @llvm.amdgcn.asyncmark(i32 6)
  call void @llvm.amdgcn.asyncmark(i32 7)
  call void @llvm.amdgcn.asyncmark(i32 8)
  call void @llvm.amdgcn.asyncmark(i32 9)
  call void @llvm.amdgcn.asyncmark(i32 10)
  ret void
}

;--- reserved-wait.ll
; RESERVED-WAIT: immarg value 4 for arg 1 out of range set
define amdgpu_kernel void @wait_asyncmark_reserved() {
  call void @llvm.amdgcn.wait.asyncmark(i16 0, i32 4)
  ret void
}
