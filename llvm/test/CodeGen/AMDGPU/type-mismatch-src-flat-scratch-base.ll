; RUN: split-file %s %t

; RUN: not --crash llc -global-isel=0 -mtriple=amdgpu12.50-amd-amdhsa < %t/lo.ll 2>&1 | FileCheck --check-prefix=LO %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgpu12.50-amd-amdhsa < %t/lo.ll 2>&1 | FileCheck --check-prefix=LO %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgpu12.50-amd-amdhsa < %t/hi.ll 2>&1 | FileCheck --check-prefix=HI %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgpu12.50-amd-amdhsa < %t/hi.ll 2>&1 | FileCheck --check-prefix=HI %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgpu12.50-amd-amdhsa < %t/combined.ll 2>&1 | FileCheck --check-prefix=COMBINED %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgpu12.50-amd-amdhsa < %t/combined.ll 2>&1 | FileCheck --check-prefix=COMBINED %s

; RUN: not --crash llc -global-isel=0 -mtriple=amdgpu12.50-amd-amdhsa < %t/write-lo.ll 2>&1 | FileCheck --check-prefix=WRITE-LO %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgpu12.50-amd-amdhsa < %t/write-lo.ll 2>&1 | FileCheck --check-prefix=WRITE-LO %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgpu12.50-amd-amdhsa < %t/write-hi.ll 2>&1 | FileCheck --check-prefix=WRITE-HI %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgpu12.50-amd-amdhsa < %t/write-hi.ll 2>&1 | FileCheck --check-prefix=WRITE-HI %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgpu12.50-amd-amdhsa < %t/write-combined.ll 2>&1 | FileCheck --check-prefix=WRITE-COMBINED %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgpu12.50-amd-amdhsa < %t/write-combined.ll 2>&1 | FileCheck --check-prefix=WRITE-COMBINED %s

; Requesting src_flat_scratch_base and its variants as a wrong type is expected to error out.

;--- lo.ll
declare i64 @llvm.read_register.i64(metadata) #0

; LO: LLVM ERROR: invalid type for register "src_flat_scratch_base_lo".
define amdgpu_kernel void @test_read_src_flat_scratch_base_lo_wrong_type(ptr addrspace(1) %out) #0 {
  %v = call i64 @llvm.read_register.i64(metadata !0)
  store i64 %v, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind }
!0 = !{!"src_flat_scratch_base_lo"}

;--- hi.ll
declare i64 @llvm.read_register.i64(metadata) #0

; HI: LLVM ERROR: invalid type for register "src_flat_scratch_base_hi".
define amdgpu_kernel void @test_read_src_flat_scratch_base_hi_wrong_type(ptr addrspace(1) %out) #0 {
  %v = call i64 @llvm.read_register.i64(metadata !0)
  store i64 %v, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind }
!0 = !{!"src_flat_scratch_base_hi"}

;--- combined.ll
declare i32 @llvm.read_register.i32(metadata) #0

; COMBINED: LLVM ERROR: invalid type for register "src_flat_scratch_base".
define amdgpu_kernel void @test_read_src_flat_scratch_base_wrong_type(ptr addrspace(1) %out) #0 {
  %v = call i32 @llvm.read_register.i32(metadata !0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind }
!0 = !{!"src_flat_scratch_base"}

;--- write-lo.ll
declare void @llvm.write_register.i64(metadata, i64) #0

; WRITE-LO: LLVM ERROR: invalid type for register "src_flat_scratch_base_lo".
define amdgpu_kernel void @test_write_src_flat_scratch_base_lo_wrong_type(i64 %v) #0 {
  call void @llvm.write_register.i64(metadata !0, i64 %v)
  ret void
}

attributes #0 = { nounwind }
!0 = !{!"src_flat_scratch_base_lo"}

;--- write-hi.ll
declare void @llvm.write_register.i64(metadata, i64) #0

; WRITE-HI: LLVM ERROR: invalid type for register "src_flat_scratch_base_hi".
define amdgpu_kernel void @test_write_src_flat_scratch_base_hi_wrong_type(i64 %v) #0 {
  call void @llvm.write_register.i64(metadata !0, i64 %v)
  ret void
}

attributes #0 = { nounwind }
!0 = !{!"src_flat_scratch_base_hi"}

;--- write-combined.ll
declare void @llvm.write_register.i32(metadata, i32) #0

; WRITE-COMBINED: LLVM ERROR: invalid type for register "src_flat_scratch_base".
define amdgpu_kernel void @test_write_src_flat_scratch_base_wrong_type(i32 %v) #0 {
  call void @llvm.write_register.i32(metadata !0, i32 %v)
  ret void
}

attributes #0 = { nounwind }
!0 = !{!"src_flat_scratch_base"}
