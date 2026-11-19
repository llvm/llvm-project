; RUN: not llc -global-isel=0 -mtriple=amdgpu12.50 -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.50 -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error %s

; src_flat_scratch_base{,_lo,_hi} are read-only source operands. They can be
; read with llvm.read_register, but they cannot be encoded as an instruction
; destination, so writing them is an error.

; CHECK: error: <unknown>:0:0: cannot write to read-only register "src_flat_scratch_base_lo" for llvm.write_register
define void @write_imm_into_src_flat_scratch_base_lo() {
  call void @llvm.write_register.i32(metadata !0, i32 0)
  ret void
}

; CHECK: error: <unknown>:0:0: cannot write to read-only register "src_flat_scratch_base_hi" for llvm.write_register
define void @write_imm_into_src_flat_scratch_base_hi() {
  call void @llvm.write_register.i32(metadata !1, i32 0)
  ret void
}

; CHECK: error: <unknown>:0:0: cannot write to read-only register "src_flat_scratch_base" for llvm.write_register
define void @write_imm_into_src_flat_scratch_base() {
  call void @llvm.write_register.i64(metadata !2, i64 0)
  ret void
}

; CHECK: error: <unknown>:0:0: cannot write to read-only register "src_flat_scratch_base_lo" for llvm.write_register
define void @write_sgpr_into_src_flat_scratch_base_lo(i32 inreg %x) {
  call void @llvm.write_register.i32(metadata !0, i32 %x)
  ret void
}

; CHECK: error: <unknown>:0:0: cannot write to read-only register "src_flat_scratch_base_hi" for llvm.write_register
define void @write_sgpr_into_src_flat_scratch_base_hi(i32 inreg %x) {
  call void @llvm.write_register.i32(metadata !1, i32 %x)
  ret void
}

; CHECK: error: <unknown>:0:0: cannot write to read-only register "src_flat_scratch_base" for llvm.write_register
define void @write_sgpr_into_src_flat_scratch_base(i64 inreg %x) {
  call void @llvm.write_register.i64(metadata !2, i64 %x)
  ret void
}

; CHECK: error: <unknown>:0:0: cannot write to read-only register "src_flat_scratch_base_lo" for llvm.write_register
define void @write_vgpr_into_src_flat_scratch_base_lo(i32 %x) {
  call void @llvm.write_register.i32(metadata !0, i32 %x)
  ret void
}

; CHECK: error: <unknown>:0:0: cannot write to read-only register "src_flat_scratch_base_hi" for llvm.write_register
define void @write_vgpr_into_src_flat_scratch_base_hi(i32 %x) {
  call void @llvm.write_register.i32(metadata !1, i32 %x)
  ret void
}

; CHECK: error: <unknown>:0:0: cannot write to read-only register "src_flat_scratch_base" for llvm.write_register
define void @write_vgpr_into_src_flat_scratch_base(i64 %x) {
  call void @llvm.write_register.i64(metadata !2, i64 %x)
  ret void
}

; CHECK: error: <unknown>:0:0: cannot write to read-only register "src_flat_scratch_base_lo" for llvm.write_register
define void @write_uniform_vgpr_into_src_flat_scratch_base_lo(ptr addrspace(3) inreg %ptr) {
  %load = load i32, ptr addrspace(3) %ptr
  call void @llvm.write_register.i32(metadata !0, i32 %load)
  ret void
}

; CHECK: error: <unknown>:0:0: cannot write to read-only register "src_flat_scratch_base_hi" for llvm.write_register
define void @write_uniform_vgpr_into_src_flat_scratch_base_hi(ptr addrspace(3) inreg %ptr) {
  %load = load i32, ptr addrspace(3) %ptr
  call void @llvm.write_register.i32(metadata !1, i32 %load)
  ret void
}

; CHECK: error: <unknown>:0:0: cannot write to read-only register "src_flat_scratch_base" for llvm.write_register
define void @write_uniform_vgpr_into_src_flat_scratch_base(ptr addrspace(3) inreg %ptr) {
  %load = load i64, ptr addrspace(3) %ptr
  call void @llvm.write_register.i64(metadata !2, i64 %load)
  ret void
}

!0 = !{!"src_flat_scratch_base_lo"}
!1 = !{!"src_flat_scratch_base_hi"}
!2 = !{!"src_flat_scratch_base"}
