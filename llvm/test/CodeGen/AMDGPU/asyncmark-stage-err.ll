; RUN: split-file %s %t
; RUN: not llc -global-isel=0 -mtriple=amdgpu12.50 -filetype=null %t/oob.ll 2>&1 | FileCheck %s -check-prefix=OOB
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.50 -filetype=null %t/oob.ll 2>&1 | FileCheck %s -check-prefix=OOB
; RUN: not llc -global-isel=0 -mtriple=amdgpu12.50 -filetype=null %t/oob-wait.ll 2>&1 | FileCheck %s -check-prefix=OOB-WAIT
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.50 -filetype=null %t/oob-wait.ll 2>&1 | FileCheck %s -check-prefix=OOB-WAIT
; RUN: not llc -global-isel=0 -mtriple=amdgpu12.50 -filetype=null %t/reserved.ll 2>&1 | FileCheck %s -check-prefix=RESERVED
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.50 -filetype=null %t/reserved.ll 2>&1 | FileCheck %s -check-prefix=RESERVED
; RUN: not llc -global-isel=0 -mtriple=amdgpu12.50 -filetype=null %t/reserved-wait.ll 2>&1 | FileCheck %s -check-prefix=RESERVED-WAIT
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.50 -filetype=null %t/reserved-wait.ll 2>&1 | FileCheck %s -check-prefix=RESERVED-WAIT

; Invalid asyncmark stages are rejected during instruction selection, with the
; same wording from both selectors. SIInsertWaitcnts must never see them.

;--- oob.ll
; Values above ALL, and values between STAGE_LAST and ALL, are out of range.
; OOB: error: {{.*}}intrinsic @llvm.amdgcn.*.asyncmark: invalid stage
define amdgpu_kernel void @asyncmark_out_of_range() {
  call void @llvm.amdgcn.asyncmark(i32 11)
  call void @llvm.amdgcn.asyncmark(i32 17)
  ret void
}

;--- oob-wait.ll
; OOB-WAIT: error: {{.*}}intrinsic @llvm.amdgcn.*.asyncmark: invalid stage
define amdgpu_kernel void @wait_asyncmark_out_of_range() {
  call void @llvm.amdgcn.wait.asyncmark(i16 0, i32 15)
  ret void
}

;--- reserved.ll
; Reserved stages are rejected too.
; RESERVED: error: {{.*}}intrinsic @llvm.amdgcn.*.asyncmark: invalid stage
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
; RESERVED-WAIT: error: {{.*}}intrinsic @llvm.amdgcn.*.asyncmark: invalid stage
define amdgpu_kernel void @wait_asyncmark_reserved() {
  call void @llvm.amdgcn.wait.asyncmark(i16 0, i32 4)
  ret void
}
