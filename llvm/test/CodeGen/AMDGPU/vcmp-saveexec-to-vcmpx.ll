; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX1010 %s
; RUN: llc -march=amdgcn -mcpu=gfx1030 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX1030 %s

; GCN-LABEL: {{^}}test_insert_vcmpx_pattern_lt:
; GFX1010: v_cmp_lt_i32_e32 vcc_lo, 15, v{{.*}}
; GFX1010-NEXT: s_and_saveexec_b32 s{{.*}}, vcc_lo
; GFX1030: s_mov_b32 s{{.*}}, exec_lo
; GFX1030-NEXT: v_cmpx_lt_i32_e32 15, v{{.*}}
define i32 @test_insert_vcmpx_pattern_lt(i32 %x) {
entry:
  %bc = icmp slt i32 %x, 16
  br i1 %bc, label %endif, label %if

if:
  %ret = shl i32 %x, 2
  ret i32 %ret

endif:
  ret i32 %x
}

; GCN-LABEL: {{^}}test_insert_vcmpx_pattern_gt:
; GFX1010: v_cmp_gt_i32_e32 vcc_lo, 17, v{{.*}}
; GFX1010-NEXT: s_and_saveexec_b32 s{{.*}}, vcc_lo
; GFX1030: s_mov_b32 s{{.*}}, exec_lo
; GFX1030-NEXT: v_cmpx_gt_i32_e32 17, v{{.*}}
define i32 @test_insert_vcmpx_pattern_gt(i32 %x) {
entry:
  %bc = icmp sgt i32 %x, 16
  br i1 %bc, label %endif, label %if

if:
  %ret = shl i32 %x, 2
  ret i32 %ret

endif:
  ret i32 %x
}

; GCN-LABEL: {{^}}test_insert_vcmpx_pattern_eq:
; GFX1010: v_cmp_ne_u32_e32 vcc_lo, 16, v{{.*}}
; GFX1010-NEXT: s_and_saveexec_b32 s{{.*}}, vcc_lo
; GFX1030: s_mov_b32 s{{.*}}, exec_lo
; GFX1030-NEXT: v_cmpx_ne_u32_e32 16, v{{.*}}
define i32 @test_insert_vcmpx_pattern_eq(i32 %x) {
entry:
  %bc = icmp eq i32 %x, 16
  br i1 %bc, label %endif, label %if

if:
  %ret = shl i32 %x, 2
  ret i32 %ret

endif:
  ret i32 %x
}

; GCN-LABEL: {{^}}test_insert_vcmpx_pattern_ne:
; GFX1010: v_cmp_eq_u32_e32 vcc_lo, 16, v{{.*}}
; GFX1010-NEXT: s_and_saveexec_b32 s{{.*}}, vcc_lo
; GFX1030: s_mov_b32 s{{.*}}, exec_lo
; GFX1030-NEXT: v_cmpx_eq_u32_e32 16, v{{.*}}
define i32 @test_insert_vcmpx_pattern_ne(i32 %x) {
entry:
  %bc = icmp ne i32 %x, 16
  br i1 %bc, label %endif, label %if

if:
  %ret = shl i32 %x, 2
  ret i32 %ret

endif:
  ret i32 %x
}

; GCN-LABEL: {{^}}test_insert_vcmpx_pattern_le:
; GFX1010: v_cmp_lt_i32_e32 vcc_lo, 16, v{{.*}}
; GFX1010-NEXT: s_and_saveexec_b32 s{{.*}}, vcc_lo
; GFX1030: s_mov_b32 s{{.*}}, exec_lo
; GFX1030-NEXT: v_cmpx_lt_i32_e32 16, v{{.*}}
define i32 @test_insert_vcmpx_pattern_le(i32 %x) {
entry:
  %bc = icmp sle i32 %x, 16
  br i1 %bc, label %endif, label %if

if:
  %ret = shl i32 %x, 2
  ret i32 %ret

endif:
  ret i32 %x
}

; GCN-LABEL: {{^}}test_insert_vcmpx_pattern_ge:
; GFX1010: v_cmp_gt_i32_e32 vcc_lo, 16, v{{.*}}
; GFX1010-NEXT: s_and_saveexec_b32 s{{.*}}, vcc_lo
; GFX1030: s_mov_b32 s{{.*}}, exec_lo
; GFX1030-NEXT: v_cmpx_gt_i32_e32 16, v{{.*}}
define i32 @test_insert_vcmpx_pattern_ge(i32 %x) {
entry:
  %bc = icmp sge i32 %x, 16
  br i1 %bc, label %endif, label %if

if:
  %ret = shl i32 %x, 2
  ret i32 %ret

endif:
  ret i32 %x
}

declare amdgpu_gfx void @check_live_outs_helper(i64) #0

; In cases where the output operand cannot be safely removed,
; don't apply the v_cmpx transformation.

; GCN-LABEL: {{^}}check_live_outs:
; GFX1010: v_cmp_eq_u32_e64 s{{.*}}, v{{.*}}, v{{.*}}
; GFX1010: s_and_saveexec_b32 s{{.*}}, s{{.*}}
; GFX1030: v_cmp_eq_u32_e64 s{{.*}}, v{{.*}}, v{{.*}}
; GFX1030: s_and_saveexec_b32 s{{.*}}, s{{.*}}
define amdgpu_cs void @check_live_outs(i32 %a, i32 %b) {
  %cond = icmp eq i32 %a, %b
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %a, i32 %b, i32 32)
  br i1 %cond, label %l1, label %l2
l1:
  call amdgpu_gfx void @check_live_outs_helper(i64 %result)
  br label %l2
l2:
  ret void
}

; Omit the transformation if the s_and_saveexec instruction overwrites
; any of the v_cmp source operands.

; GCN-LABEL: check_saveexec_overwrites_vcmp_source:
; GCN:  %bb.1: ; %then
; GFX1010:          v_cmp_ge_i32_e32 vcc_lo, s[[A:[0-9]+]], v{{.*}}
; GFX1010-NEXT:     v_mov_b32_e32 {{.*}}, s[[A]]
; GFX1010-NEXT:     s_and_saveexec_b32 s[[A]], vcc_lo
; GFX1030:          v_cmp_ge_i32_e32 vcc_lo, s[[A:[0-9]+]], v{{.*}}
; GFX1030-NEXT:     v_mov_b32_e32 {{.*}}, s[[A]]
; GFX1030-NEXT:     s_and_saveexec_b32 s[[A]], vcc_lo
define i32 @check_saveexec_overwrites_vcmp_source(i32 inreg %a, i32 inreg %b) {
entry:
  %0 = icmp sge i32 %a, 0
  br i1 %0, label %if, label %then

if:
  %1 = shl i32 %a, 2
  %2 = or i32 %1, %b
  ret i32 %2

then:
  %3 = call i64 @llvm.amdgcn.icmp.i32(i32 %a, i32 %b, i32 32)
  %4 = trunc i64 %3 to i32
  %5 = icmp slt i32 %4, %b
  br i1 %5, label %after, label %end

after:
  ret i32 %4

end:
  ret i32 %a
}

declare i64 @llvm.amdgcn.icmp.i32(i32, i32, i32) #0
