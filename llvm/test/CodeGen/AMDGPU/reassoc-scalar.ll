; RUN: llc -mtriple=amdgpu8.02 < %s | FileCheck -check-prefixes=GCN,GFX8 %s
; RUN: llc -mtriple=amdgpu9.00 < %s | FileCheck -check-prefixes=GCN,GFX9 %s

; GCN-LABEL: reassoc_i32:
; GCN: s_add_i32 [[ADD1:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GFX8: v_add_u32_e32 v{{[0-9]+}}, vcc, [[ADD1]], v{{[0-9]+}}
; GFX9: v_add_u32_e32 v{{[0-9]+}}, [[ADD1]], v{{[0-9]+}}
define amdgpu_kernel void @reassoc_i32(ptr addrspace(1) %arg, i32 %x, i32 %y) {
bb:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %add1 = add i32 %x, %tid
  %add2 = add i32 %add1, %y
  store i32 %add2, ptr addrspace(1) %arg, align 4
  ret void
}

; GCN-LABEL: reassoc_i32_swap_arg_order:
; GCN:  s_add_i32 [[ADD1:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GFX8: v_add_u32_e32 v{{[0-9]+}}, vcc, [[ADD1]], v{{[0-9]+}}
; GFX9: v_add_u32_e32 v{{[0-9]+}}, [[ADD1]], v{{[0-9]+}}
define amdgpu_kernel void @reassoc_i32_swap_arg_order(ptr addrspace(1) %arg, i32 %x, i32 %y) {
bb:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %add1 = add i32 %tid, %x
  %add2 = add i32 %y, %add1
  store i32 %add2, ptr addrspace(1) %arg, align 4
  ret void
}

; GCN-LABEL: reassoc_i64:
; GCN:      s_add_u32 [[ADD1L:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GCN:      s_addc_u32 [[ADD1H:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GFX8-DAG: v_add_u32_e32 v{{[0-9]+}}, vcc, [[ADD1L]], v{{[0-9]+}}
; GFX9-DAG: v_add_co_u32_e32 v{{[0-9]+}}, vcc, [[ADD1L]], v{{[0-9]+}}
; GCN-DAG:  v_mov_b32_e32 [[VADD1H:v[0-9]+]], [[ADD1H]]
; GFX8:     v_addc_u32_e32 v{{[0-9]+}}, vcc, 0, [[VADD1H]], vcc
; GFX9:     v_addc_co_u32_e32 v{{[0-9]+}}, vcc, 0, [[VADD1H]], vcc
define amdgpu_kernel void @reassoc_i64(ptr addrspace(1) %arg, i64 %x, i64 %y) {
bb:
  %tid32 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tid = zext i32 %tid32 to i64
  %add1 = add i64 %x, %tid
  %add2 = add i64 %add1, %y
  store i64 %add2, ptr addrspace(1) %arg, align 8
  ret void
}

; GCN-LABEL: reassoc_v2i32:
; GCN: s_add_i32 [[ADD1:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GCN-DAG: s_add_i32 [[ADD2:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GFX8-DAG: v_add_u32_e32 v{{[0-9]+}}, vcc, [[ADD1]], v{{[0-9]+}}
; GFX8: v_add_u32_e32 v{{[0-9]+}}, vcc, [[ADD2]], v{{[0-9]+}}
; GFX9-DAG: v_add_u32_e32 v{{[0-9]+}}, [[ADD1]], v{{[0-9]+}}
; GFX9: v_add_u32_e32 v{{[0-9]+}}, [[ADD2]], v{{[0-9]+}}
define amdgpu_kernel void @reassoc_v2i32(ptr addrspace(1) %arg, <2 x i32> %x, <2 x i32> %y) {
bb:
  %t1 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %t2 = tail call i32 @llvm.amdgcn.workitem.id.y()
  %v1 = insertelement <2 x i32> poison, i32 %t1, i32 0
  %v2 = insertelement <2 x i32> %v1, i32 %t2, i32 1
  %add1 = add <2 x i32> %x, %v2
  %add2 = add <2 x i32> %add1, %y
  store <2 x i32> %add2, ptr addrspace(1) %arg, align 4
  ret void
}

; GCN-LABEL: reassoc_i32_nuw:
; GCN:  s_add_i32 [[ADD1:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GFX8: v_add_u32_e32 v{{[0-9]+}}, vcc, [[ADD1]], v{{[0-9]+}}
; GFX9: v_add_u32_e32 v{{[0-9]+}}, [[ADD1]], v{{[0-9]+}}
define amdgpu_kernel void @reassoc_i32_nuw(ptr addrspace(1) %arg, i32 %x, i32 %y) {
bb:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %add1 = add i32 %x, %tid
  %add2 = add nuw i32 %add1, %y
  store i32 %add2, ptr addrspace(1) %arg, align 4
  ret void
}

; GCN-LABEL: reassoc_i32_multiuse:
; GFX8: v_add_u32_e32 [[ADD1:v[0-9]+]], vcc, s{{[0-9]+}}, v{{[0-9]+}}
; GFX9: v_add_u32_e32 [[ADD1:v[0-9]+]], s{{[0-9]+}}, v{{[0-9]+}}
; GFX8: v_add_u32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, [[ADD1]]
; GFX9: v_add_u32_e32 v{{[0-9]+}}, s{{[0-9]+}}, [[ADD1]]
define amdgpu_kernel void @reassoc_i32_multiuse(ptr addrspace(1) %arg, i32 %x, i32 %y) {
bb:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %add1 = add i32 %x, %tid
  %add2 = add i32 %add1, %y
  store volatile i32 %add1, ptr addrspace(1) %arg, align 4
  store volatile i32 %add2, ptr addrspace(1) %arg, align 4
  ret void
}

; TODO: This should be reassociated as well, however it is disabled to avoid endless
;       loop since DAGCombiner::ReassociateOps() reverts the reassociation.
; GCN-LABEL: reassoc_i32_const:
; GFX8: v_add_u32_e32 [[ADD1:v[0-9]+]], vcc, 42, v{{[0-9]+}}
; GFX9: v_add_u32_e32 [[ADD1:v[0-9]+]],  42, v{{[0-9]+}}
; GFX8: v_add_u32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, [[ADD1]]
; GFX9: v_add_u32_e32 v{{[0-9]+}}, s{{[0-9]+}}, [[ADD1]]
define amdgpu_kernel void @reassoc_i32_const(ptr addrspace(1) %arg, i32 %x) {
bb:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %add1 = add i32 %tid, 42
  %add2 = add i32 %add1, %x
  store volatile i32 %add1, ptr addrspace(1) %arg, align 4
  store volatile i32 %add2, ptr addrspace(1) %arg, align 4
  ret void
}

@var = common hidden local_unnamed_addr addrspace(1) global [4 x i32] zeroinitializer, align 4

; GCN-LABEL: reassoc_i32_ga:
; GCN: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, var@rel32@lo+4
; GCN: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, var@rel32@hi+12
; GCN: s_endpgm
define amdgpu_kernel void @reassoc_i32_ga(i64 %x) {
bb:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %t64 = zext i32 %tid to i64
  %add1 = getelementptr [4 x i32], ptr addrspace(1) @var, i64 0, i64 %t64
  %add2 = getelementptr i32, ptr addrspace(1) %add1, i64 %x
  store volatile i32 1, ptr addrspace(1) %add2, align 4
  ret void
}

; Multiply: (x * tid) * y -- backend does NOT reassociate mul to scalar.
; Both muls use VALU even though x and y are uniform.
; The reassociate pass groups the uniform operands (x * y) so this
; lowers to s_mul + v_mul; see reassoc-uniform-e2e.ll.
; GCN-LABEL: reassoc_mul_i32:
; GCN-NOT: s_mul
; GCN: v_mul_lo_u32
; GCN: v_mul_lo_u32
define void @reassoc_mul_i32(ptr addrspace(1) inreg %arg, i32 inreg %x, i32 inreg %y) {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %mul1 = mul i32 %x, %tid
  %mul2 = mul i32 %mul1, %y
  store i32 %mul2, ptr addrspace(1) %arg, align 4
  ret void
}

; Multiply with uniforms pre-grouped: (x * y) * tid -- produces s_mul + v_mul.
; GCN-LABEL: reassoc_mul_i32_uniform_grouped:
; GCN: s_mul_i32 [[MUL1:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GCN: v_mul_lo_u32 v{{[0-9]+}}, [[MUL1]], v{{[0-9]+}}
define void @reassoc_mul_i32_uniform_grouped(ptr addrspace(1) inreg %arg, i32 inreg %x, i32 inreg %y) {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %mul1 = mul i32 %x, %y
  %mul2 = mul i32 %mul1, %tid
  store i32 %mul2, ptr addrspace(1) %arg, align 4
  ret void
}

; OR: (x | tid) | y -- backend does NOT reassociate or to scalar.
; Both ors use VALU.
; The reassociate pass groups the uniform operands (x | y) so this
; lowers to s_or + v_or; see reassoc-uniform-e2e.ll.
; GCN-LABEL: reassoc_or_i32:
; GCN-NOT: s_or
; GCN: v_or_b32_e32
; GCN: v_or_b32_e32
define void @reassoc_or_i32(ptr addrspace(1) inreg %arg, i32 inreg %x, i32 inreg %y) {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %or1 = or i32 %x, %tid
  %or2 = or i32 %or1, %y
  store i32 %or2, ptr addrspace(1) %arg, align 4
  ret void
}

; OR with uniforms pre-grouped: (x | y) | tid -- produces s_or + v_or.
; GCN-LABEL: reassoc_or_i32_uniform_grouped:
; GCN: s_or_b32 [[OR1:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GCN: v_or_b32_e32 v{{[0-9]+}}, [[OR1]], v{{[0-9]+}}
define void @reassoc_or_i32_uniform_grouped(ptr addrspace(1) inreg %arg, i32 inreg %x, i32 inreg %y) {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %or1 = or i32 %x, %y
  %or2 = or i32 %or1, %tid
  store i32 %or2, ptr addrspace(1) %arg, align 4
  ret void
}

; AND: (x & tid) & y -- backend does NOT reassociate and to scalar.
; Both ands use VALU.
; The reassociate pass groups the uniform operands (x & y) so this
; lowers to s_and + v_and; see reassoc-uniform-e2e.ll.
; GCN-LABEL: reassoc_and_i32:
; GCN-NOT: s_and
; GCN: v_and_b32_e32
; GCN: v_and_b32_e32
define void @reassoc_and_i32(ptr addrspace(1) inreg %arg, i32 inreg %x, i32 inreg %y) {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %and1 = and i32 %x, %tid
  %and2 = and i32 %and1, %y
  store i32 %and2, ptr addrspace(1) %arg, align 4
  ret void
}

; AND with uniforms pre-grouped: (x & y) & tid -- produces s_and + v_and.
; GCN-LABEL: reassoc_and_i32_uniform_grouped:
; GCN: s_and_b32 [[AND1:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GCN: v_and_b32_e32 v{{[0-9]+}}, [[AND1]], v{{[0-9]+}}
define void @reassoc_and_i32_uniform_grouped(ptr addrspace(1) inreg %arg, i32 inreg %x, i32 inreg %y) {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %and1 = and i32 %x, %y
  %and2 = and i32 %and1, %tid
  store i32 %and2, ptr addrspace(1) %arg, align 4
  ret void
}

; XOR: (x ^ tid) ^ y -- backend ALREADY reassociates xor to scalar.
; The DAG combiner handles this, producing s_xor + v_xor for both variants.
; GCN-LABEL: reassoc_xor_i32:
; GCN: s_xor_b32 [[XOR1:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GCN: v_xor_b32_e32 v{{[0-9]+}}, [[XOR1]], v{{[0-9]+}}
define void @reassoc_xor_i32(ptr addrspace(1) inreg %arg, i32 inreg %x, i32 inreg %y) {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %xor1 = xor i32 %x, %tid
  %xor2 = xor i32 %xor1, %y
  store i32 %xor2, ptr addrspace(1) %arg, align 4
  ret void
}

; Multi-use: intermediate (x * tid) is stored, blocks reassociation.
; Both muls use VALU. The reassociate pass cannot group the uniforms here
; because the intermediate result is used more than once.
; GCN-LABEL: reassoc_mul_i32_multiuse:
; GCN-NOT: s_mul
; GCN: v_mul_lo_u32
; GCN: v_mul_lo_u32
define void @reassoc_mul_i32_multiuse(ptr addrspace(1) inreg %arg, i32 inreg %x, i32 inreg %y) {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %mul1 = mul i32 %x, %tid
  %mul2 = mul i32 %mul1, %y
  store volatile i32 %mul1, ptr addrspace(1) %arg, align 4
  store volatile i32 %mul2, ptr addrspace(1) %arg, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workitem.id.y()
