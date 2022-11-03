; RUN: opt -S -codegenprepare -mtriple=amdgcn-unknown-unknown -mcpu=tahiti < %s | FileCheck -check-prefix=OPT -check-prefix=OPT-SI -check-prefix=OPT-SICIVI %s
; RUN: opt -S -codegenprepare -mtriple=amdgcn-unknown-unknown -mcpu=bonaire < %s | FileCheck -check-prefix=OPT -check-prefix=OPT-CI -check-prefix=OPT-SICIVI %s
; RUN: opt -S -codegenprepare -mtriple=amdgcn-unknown-unknown -mcpu=tonga -mattr=-flat-for-global < %s | FileCheck -check-prefix=OPT -check-prefix=OPT-VI -check-prefix=OPT-SICIVI %s
; RUN: opt -S -codegenprepare -mtriple=amdgcn-unknown-unknown -mcpu=gfx900 < %s | FileCheck -check-prefix=OPT -check-prefix=OPT-GFX9 %s
; RUN: llc -march=amdgcn -mcpu=tahiti -mattr=-promote-alloca -amdgpu-scalarize-global-loads=false -amdgpu-sroa=0 < %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=SICIVI %s
; RUN: llc -march=amdgcn -mcpu=bonaire -mattr=-promote-alloca -amdgpu-scalarize-global-loads=false -amdgpu-sroa=0 < %s | FileCheck -check-prefix=GCN -check-prefix=CI -check-prefix=SICIVI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -amdgpu-scalarize-global-loads=false -mattr=-promote-alloca -amdgpu-sroa=0 < %s | FileCheck -check-prefix=GCN -check-prefix=VI -check-prefix=SICIVI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-promote-alloca -amdgpu-scalarize-global-loads=false -amdgpu-sroa=0 < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

; OPT-LABEL: @test_sink_global_small_offset_i32(
; OPT-CI-NOT: getelementptr i32, ptr addrspace(1) %in
; OPT-VI: getelementptr i32, ptr addrspace(1) %in
; OPT: br i1
; OPT-CI: getelementptr i8,

; GCN-LABEL: {{^}}test_sink_global_small_offset_i32:
define amdgpu_kernel void @test_sink_global_small_offset_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %out.gep = getelementptr i32, ptr addrspace(1) %out, i64 999999
  %in.gep = getelementptr i32, ptr addrspace(1) %in, i64 7
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i32, ptr addrspace(1) %in.gep
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(1) %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_sink_global_small_max_i32_ds_offset(
; OPT: %in.gep = getelementptr i8, ptr addrspace(1) %in, i64 65535
; OPT: br i1

; GCN-LABEL: {{^}}test_sink_global_small_max_i32_ds_offset:
; GCN: s_and_saveexec_b64
; SICIVI: buffer_load_sbyte {{v[0-9]+}}, off, {{s\[[0-9]+:[0-9]+\]}}, s{{[0-9]+$}}

; GFX9: v_mov_b32_e32 [[VOFFSET:v[0-9]+]], 0xf000{{$}}
; GFX9: global_load_sbyte {{v[0-9]+}}, [[VOFFSET]], {{s\[[0-9]+:[0-9]+\]}} offset:4095{{$}}
; GCN: {{^}}.LBB1_2:
; GCN: s_or_b64 exec
define amdgpu_kernel void @test_sink_global_small_max_i32_ds_offset(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %out.gep = getelementptr i32, ptr addrspace(1) %out, i64 99999
  %in.gep = getelementptr i8, ptr addrspace(1) %in, i64 65535
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i8, ptr addrspace(1) %in.gep
  %tmp2 = sext i8 %tmp1 to i32
  br label %endif

endif:
  %x = phi i32 [ %tmp2, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(1) %out.gep
  br label %done

done:
  ret void
}

; GCN-LABEL: {{^}}test_sink_global_small_max_mubuf_offset:
; GCN: s_and_saveexec_b64
; SICIVI: buffer_load_sbyte {{v[0-9]+}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 offset:4095{{$}}
; GFX9: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; GFX9: global_load_sbyte {{v[0-9]+}}, [[ZERO]], {{s\[[0-9]+:[0-9]+\]}} offset:4095{{$}}
; GCN: {{^}}.LBB2_2:
; GCN: s_or_b64 exec
define amdgpu_kernel void @test_sink_global_small_max_mubuf_offset(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %out.gep = getelementptr i32, ptr addrspace(1) %out, i32 1024
  %in.gep = getelementptr i8, ptr addrspace(1) %in, i64 4095
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i8, ptr addrspace(1) %in.gep
  %tmp2 = sext i8 %tmp1 to i32
  br label %endif

endif:
  %x = phi i32 [ %tmp2, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(1) %out.gep
  br label %done

done:
  ret void
}

; GCN-LABEL: {{^}}test_sink_global_small_max_plus_1_mubuf_offset:
; GCN: s_and_saveexec_b64
; SICIVI: buffer_load_sbyte {{v[0-9]+}}, off, {{s\[[0-9]+:[0-9]+\]}}, s{{[0-9]+$}}
; GFX9: v_mov_b32_e32 [[VOFFSET:v[0-9]+]], 0x1000{{$}}
; GFX9: global_load_sbyte {{v[0-9]+}}, [[VOFFSET]], {{s\[[0-9]+:[0-9]+\]$}}
; GCN: {{^}}.LBB3_2:
; GCN: s_or_b64 exec
define amdgpu_kernel void @test_sink_global_small_max_plus_1_mubuf_offset(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %out.gep = getelementptr i32, ptr addrspace(1) %out, i64 99999
  %in.gep = getelementptr i8, ptr addrspace(1) %in, i64 4096
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i8, ptr addrspace(1) %in.gep
  %tmp2 = sext i8 %tmp1 to i32
  br label %endif

endif:
  %x = phi i32 [ %tmp2, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(1) %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_sink_scratch_small_offset_i32(
; OPT-NOT:  getelementptr [512 x i32]
; OPT: br i1
; OPT: getelementptr i8,

; GCN-LABEL: {{^}}test_sink_scratch_small_offset_i32:
; GCN: s_and_saveexec_b64
; GCN: buffer_store_dword {{v[0-9]+}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 offset:4092{{$}}
; GCN: buffer_load_dword {{v[0-9]+}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 offset:4092 glc{{$}}
; GCN: {{^}}.LBB4_2:
define amdgpu_kernel void @test_sink_scratch_small_offset_i32(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %arg) {
entry:
  %alloca = alloca [512 x i32], align 4, addrspace(5)
  %out.gep.0 = getelementptr i32, ptr addrspace(1) %out, i64 999998
  %out.gep.1 = getelementptr i32, ptr addrspace(1) %out, i64 999999
  %add.arg = add i32 %arg, 8
  %alloca.gep = getelementptr [512 x i32], ptr addrspace(5) %alloca, i32 0, i32 1022
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  store volatile i32 123, ptr addrspace(5) %alloca.gep
  %tmp1 = load volatile i32, ptr addrspace(5) %alloca.gep
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(1) %out.gep.0
  %load = load volatile i32, ptr addrspace(5) %alloca.gep
  store i32 %load, ptr addrspace(1) %out.gep.1
  br label %done

done:
  ret void
}

; This ends up not fitting due to the reserved 4 bytes at offset 0
; OPT-LABEL: @test_sink_scratch_small_offset_i32_reserved(
; OPT-NOT:  getelementptr [512 x i32]
; OPT: br i1
; OPT: getelementptr i8,

; GCN-LABEL: {{^}}test_sink_scratch_small_offset_i32_reserved:
; GCN: s_and_saveexec_b64
; GCN: v_mov_b32_e32 [[BASE_FI0:v[0-9]+]], 4
; GCN: buffer_store_dword {{v[0-9]+}}, [[BASE_FI0]], {{s\[[0-9]+:[0-9]+\]}}, 0 offen offset:4092{{$}}
; GCN: v_mov_b32_e32 [[BASE_FI1:v[0-9]+]], 4
; GCN: buffer_load_dword {{v[0-9]+}}, [[BASE_FI1]], {{s\[[0-9]+:[0-9]+\]}}, 0 offen offset:4092 glc{{$}}
; GCN: {{^.LBB[0-9]+}}_2:

define amdgpu_kernel void @test_sink_scratch_small_offset_i32_reserved(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %arg) {
entry:
  %alloca = alloca [512 x i32], align 4, addrspace(5)
  %out.gep.0 = getelementptr i32, ptr addrspace(1) %out, i64 999998
  %out.gep.1 = getelementptr i32, ptr addrspace(1) %out, i64 999999
  %add.arg = add i32 %arg, 8
  %alloca.gep = getelementptr [512 x i32], ptr addrspace(5) %alloca, i32 0, i32 1023
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  store volatile i32 123, ptr addrspace(5) %alloca.gep
  %tmp1 = load volatile i32, ptr addrspace(5) %alloca.gep
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(1) %out.gep.0
  %load = load volatile i32, ptr addrspace(5) %alloca.gep
  store i32 %load, ptr addrspace(1) %out.gep.1
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_no_sink_scratch_large_offset_i32(
; OPT: %alloca.gep = getelementptr [512 x i32], ptr addrspace(5) %alloca, i32 0, i32 1024
; OPT: br i1
; OPT-NOT: ptrtoint

; GCN-LABEL: {{^}}test_no_sink_scratch_large_offset_i32:
; GCN: s_and_saveexec_b64
; GCN: buffer_store_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen{{$}}
; GCN: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen glc{{$}}
; GCN: {{^.LBB[0-9]+}}_2:
define amdgpu_kernel void @test_no_sink_scratch_large_offset_i32(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %arg) {
entry:
  %alloca = alloca [512 x i32], align 4, addrspace(5)
  %out.gep.0 = getelementptr i32, ptr addrspace(1) %out, i64 999998
  %out.gep.1 = getelementptr i32, ptr addrspace(1) %out, i64 999999
  %add.arg = add i32 %arg, 8
  %alloca.gep = getelementptr [512 x i32], ptr addrspace(5) %alloca, i32 0, i32 1024
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  store volatile i32 123, ptr addrspace(5) %alloca.gep
  %tmp1 = load volatile i32, ptr addrspace(5) %alloca.gep
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(1) %out.gep.0
  %load = load volatile i32, ptr addrspace(5) %alloca.gep
  store i32 %load, ptr addrspace(1) %out.gep.1
  br label %done

done:
  ret void
}

; GCN-LABEL: {{^}}test_sink_global_vreg_sreg_i32:
; GCN: s_and_saveexec_b64
; CI: buffer_load_dword {{v[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; VI: flat_load_dword v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
; GCN: {{^.LBB[0-9]+}}_2:
define amdgpu_kernel void @test_sink_global_vreg_sreg_i32(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %offset) {
entry:
  %offset.ext = zext i32 %offset to i64
  %out.gep = getelementptr i32, ptr addrspace(1) %out, i64 999999
  %in.gep = getelementptr i32, ptr addrspace(1) %in, i64 %offset.ext
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i32, ptr addrspace(1) %in.gep
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(1) %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_sink_constant_small_offset_i32
; OPT-NOT:  getelementptr i32, ptr addrspace(4)
; OPT: br i1

; GCN-LABEL: {{^}}test_sink_constant_small_offset_i32:
; GCN: s_and_saveexec_b64
; SI: s_load_dword s{{[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0x7{{$}}
; GCN: s_or_b64 exec, exec
define amdgpu_kernel void @test_sink_constant_small_offset_i32(ptr addrspace(1) %out, ptr addrspace(4) %in) {
entry:
  %out.gep = getelementptr i32, ptr addrspace(1) %out, i64 999999
  %in.gep = getelementptr i32, ptr addrspace(4) %in, i64 7
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i32, ptr addrspace(4) %in.gep
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(1) %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_sink_constant_max_8_bit_offset_i32
; OPT-NOT:  getelementptr i32, ptr addrspace(4)
; OPT: br i1

; GCN-LABEL: {{^}}test_sink_constant_max_8_bit_offset_i32:
; GCN: s_and_saveexec_b64
; SI: s_load_dword s{{[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0xff{{$}}
; GCN: s_or_b64 exec, exec
define amdgpu_kernel void @test_sink_constant_max_8_bit_offset_i32(ptr addrspace(1) %out, ptr addrspace(4) %in) {
entry:
  %out.gep = getelementptr i32, ptr addrspace(1) %out, i64 999999
  %in.gep = getelementptr i32, ptr addrspace(4) %in, i64 255
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i32, ptr addrspace(4) %in.gep
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(1) %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_sink_constant_max_8_bit_offset_p1_i32
; OPT-SI:  getelementptr i32, ptr addrspace(4)
; OPT-CI-NOT:  getelementptr i32, ptr addrspace(4)
; OPT-VI-NOT:  getelementptr i32, ptr addrspace(4)
; OPT: br i1

; GCN-LABEL: {{^}}test_sink_constant_max_8_bit_offset_p1_i32:
; GCN: s_and_saveexec_b64
; SI: s_movk_i32 [[OFFSET:s[0-9]+]], 0x400

; SI: s_load_dword s{{[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, [[OFFSET]]{{$}}
; GCN: s_or_b64 exec, exec
define amdgpu_kernel void @test_sink_constant_max_8_bit_offset_p1_i32(ptr addrspace(1) %out, ptr addrspace(4) %in) {
entry:
  %out.gep = getelementptr i32, ptr addrspace(1) %out, i64 999999
  %in.gep = getelementptr i32, ptr addrspace(4) %in, i64 256
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i32, ptr addrspace(4) %in.gep
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(1) %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_sink_constant_max_32_bit_offset_i32
; OPT-SI: getelementptr i32, ptr addrspace(4)
; OPT-CI-NOT: getelementptr i32, ptr addrspace(4)
; OPT: br i1

; GCN-LABEL: {{^}}test_sink_constant_max_32_bit_offset_i32:
; GCN: s_and_saveexec_b64
; SI: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, -4{{$}}
; SI: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, 3{{$}}
; SI: s_load_dword s{{[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0x0{{$}}

; VI: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, -4{{$}}
; VI: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, 3{{$}}
; VI: s_load_dword s{{[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0x0{{$}}

; CI: s_load_dword s{{[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0xffffffff{{$}}

; GCN: s_or_b64 exec, exec
define amdgpu_kernel void @test_sink_constant_max_32_bit_offset_i32(ptr addrspace(1) %out, ptr addrspace(4) %in) {
entry:
  %out.gep = getelementptr i32, ptr addrspace(1) %out, i64 999999
  %in.gep = getelementptr i32, ptr addrspace(4) %in, i64 4294967295
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i32, ptr addrspace(4) %in.gep
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(1) %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_sink_constant_max_32_bit_offset_p1_i32
; OPT: getelementptr i32, ptr addrspace(4)
; OPT: br i1

; GCN-LABEL: {{^}}test_sink_constant_max_32_bit_offset_p1_i32:
; GCN: s_and_saveexec_b64
; GCN: s_add_u32
; GCN: s_addc_u32
; SI: s_load_dword s{{[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0x0{{$}}
; GCN: s_or_b64 exec, exec
define amdgpu_kernel void @test_sink_constant_max_32_bit_offset_p1_i32(ptr addrspace(1) %out, ptr addrspace(4) %in) {
entry:
  %out.gep = getelementptr i32, ptr addrspace(1) %out, i64 999999
  %in.gep = getelementptr i32, ptr addrspace(4) %in, i64 17179869181
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i32, ptr addrspace(4) %in.gep
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(1) %out.gep
  br label %done

done:
  ret void
}

; GCN-LABEL: {{^}}test_sink_constant_max_20_bit_byte_offset_i32:
; GCN: s_and_saveexec_b64
; SI: s_mov_b32 [[OFFSET:s[0-9]+]], 0xffffc{{$}}
; SI: s_load_dword s{{[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, [[OFFSET]]{{$}}

; CI: s_load_dword s{{[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0x3ffff{{$}}
; VI: s_load_dword s{{[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0xffffc{{$}}

; GCN: s_or_b64 exec, exec
define amdgpu_kernel void @test_sink_constant_max_20_bit_byte_offset_i32(ptr addrspace(1) %out, ptr addrspace(4) %in) {
entry:
  %out.gep = getelementptr i32, ptr addrspace(1) %out, i64 999999
  %in.gep = getelementptr i32, ptr addrspace(4) %in, i64 262143
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i32, ptr addrspace(4) %in.gep
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(1) %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_sink_constant_max_20_bit_byte_offset_p1_i32
; OPT-SI: getelementptr i32, ptr addrspace(4)
; OPT-CI-NOT: getelementptr i32, ptr addrspace(4)
; OPT-VI: getelementptr i32, ptr addrspace(4)
; OPT: br i1

; GCN-LABEL: {{^}}test_sink_constant_max_20_bit_byte_offset_p1_i32:
; GCN: s_and_saveexec_b64
; SI: s_mov_b32 [[OFFSET:s[0-9]+]], 0x100000{{$}}
; SI: s_load_dword s{{[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, [[OFFSET]]{{$}}

; CI: s_load_dword s{{[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0x40000{{$}}

; VI: s_mov_b32 [[OFFSET:s[0-9]+]], 0x100000{{$}}
; VI: s_load_dword s{{[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, [[OFFSET]]{{$}}

; GCN: s_or_b64 exec, exec
define amdgpu_kernel void @test_sink_constant_max_20_bit_byte_offset_p1_i32(ptr addrspace(1) %out, ptr addrspace(4) %in) {
entry:
  %out.gep = getelementptr i32, ptr addrspace(1) %out, i64 999999
  %in.gep = getelementptr i32, ptr addrspace(4) %in, i64 262144
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i32, ptr addrspace(4) %in.gep
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(1) %out.gep
  br label %done

done:
  ret void
}

%struct.foo = type { [3 x float], [3 x float] }

; OPT-LABEL: @sink_ds_address(
; OPT: getelementptr inbounds i8,

; GCN-LABEL: {{^}}sink_ds_address:
; GCN: s_load_dword [[SREG1:s[0-9]+]],
; GCN: v_mov_b32_e32 [[VREG1:v[0-9]+]], [[SREG1]]
; GCN-DAG: ds_read2_b32 v[{{[0-9+:[0-9]+}}], [[VREG1]] offset0:3 offset1:5
define amdgpu_kernel void @sink_ds_address(ptr addrspace(3) nocapture %ptr) nounwind {
entry:
  %x = getelementptr inbounds %struct.foo, ptr addrspace(3) %ptr, i32 0, i32 1, i32 0
  %y = getelementptr inbounds %struct.foo, ptr addrspace(3) %ptr, i32 0, i32 1, i32 2
  br label %bb32

bb32:
  %a = load float, ptr addrspace(3) %x, align 4
  %b = load float, ptr addrspace(3) %y, align 4
  %cmp = fcmp one float %a, %b
  br i1 %cmp, label %bb34, label %bb33

bb33:
  unreachable

bb34:
  unreachable
}

; Address offset is not a multiple of 4. This is a valid mubuf offset,
; but not smrd.

; OPT-LABEL: @test_sink_constant_small_max_mubuf_offset_load_i32_align_1(
; OPT: br i1 %tmp0,
; OPT: if:
; OPT: getelementptr i8, {{.*}} 4095
define amdgpu_kernel void @test_sink_constant_small_max_mubuf_offset_load_i32_align_1(ptr addrspace(1) %out, ptr addrspace(4) %in) {
entry:
  %out.gep = getelementptr i32, ptr addrspace(1) %out, i32 1024
  %in.gep = getelementptr i8, ptr addrspace(4) %in, i64 4095
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i32, ptr addrspace(4) %in.gep, align 1
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(1) %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_sink_local_small_offset_atomicrmw_i32(
; OPT: %sunkaddr = getelementptr i8, ptr addrspace(3) %in, i32 28
; OPT: %tmp1 = atomicrmw add ptr addrspace(3) %sunkaddr, i32 2 seq_cst
define amdgpu_kernel void @test_sink_local_small_offset_atomicrmw_i32(ptr addrspace(3) %out, ptr addrspace(3) %in) {
entry:
  %out.gep = getelementptr i32, ptr addrspace(3) %out, i32 999999
  %in.gep = getelementptr i32, ptr addrspace(3) %in, i32 7
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = atomicrmw add ptr addrspace(3) %in.gep, i32 2 seq_cst
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(3) %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_sink_local_small_offset_cmpxchg_i32(
; OPT: %sunkaddr = getelementptr i8, ptr addrspace(3) %in, i32 28
; OPT: %tmp1.struct = cmpxchg ptr addrspace(3) %sunkaddr, i32 undef, i32 2 seq_cst monotonic
define amdgpu_kernel void @test_sink_local_small_offset_cmpxchg_i32(ptr addrspace(3) %out, ptr addrspace(3) %in) {
entry:
  %out.gep = getelementptr i32, ptr addrspace(3) %out, i32 999999
  %in.gep = getelementptr i32, ptr addrspace(3) %in, i32 7
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1.struct = cmpxchg ptr addrspace(3) %in.gep, i32 undef, i32 2 seq_cst monotonic
  %tmp1 = extractvalue { i32, i1 } %tmp1.struct, 0
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(3) %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_wrong_operand_local_small_offset_cmpxchg_i32(
; OPT: %in.gep = getelementptr i32, ptr addrspace(3) %in, i32 7
; OPT: br i1
; OPT: cmpxchg ptr addrspace(3) undef, ptr addrspace(3) %in.gep, ptr addrspace(3) undef seq_cst monotonic
define amdgpu_kernel void @test_wrong_operand_local_small_offset_cmpxchg_i32(ptr addrspace(3) %out, ptr addrspace(3) %in) {
entry:
  %out.gep = getelementptr ptr addrspace(3), ptr addrspace(3) %out, i32 999999
  %in.gep = getelementptr i32, ptr addrspace(3) %in, i32 7
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1.struct = cmpxchg ptr addrspace(3) undef, ptr addrspace(3) %in.gep, ptr addrspace(3) undef seq_cst monotonic
  %tmp1 = extractvalue { ptr addrspace(3), i1 } %tmp1.struct, 0
  br label %endif

endif:
  %x = phi ptr addrspace(3) [ %tmp1, %if ], [ null, %entry ]
  store ptr addrspace(3) %x, ptr addrspace(3) %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_sink_global_small_min_scratch_global_offset(
; OPT-SICIVI: %in.gep = getelementptr i8, ptr addrspace(1) %in, i64 -4096
; OPT-SICIV: br
; OPT-SICIVI: %tmp1 = load i8, ptr addrspace(1) %in.gep

; OPT-GFX9: br
; OPT-GFX9: %sunkaddr = getelementptr i8, ptr addrspace(1) %in, i64 -4096
; OPT-GFX9: load i8, ptr addrspace(1) %sunkaddr

; GCN-LABEL: {{^}}test_sink_global_small_min_scratch_global_offset:
; GFX9: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; GFX9: global_load_sbyte v{{[0-9]+}}, [[ZERO]], s{{\[[0-9]+:[0-9]+\]}} offset:-4096{{$}}
define amdgpu_kernel void @test_sink_global_small_min_scratch_global_offset(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %out.gep = getelementptr i32, ptr addrspace(1) %out, i32 1024
  %in.gep = getelementptr i8, ptr addrspace(1) %in, i64 -4096
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i8, ptr addrspace(1) %in.gep
  %tmp2 = sext i8 %tmp1 to i32
  br label %endif

endif:
  %x = phi i32 [ %tmp2, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(1) %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_sink_global_small_min_scratch_global_neg1_offset(
; OPT: %in.gep = getelementptr i8, ptr addrspace(1) %in, i64 -4097
; OPT: br
; OPT: load i8, ptr addrspace(1) %in.gep

; GCN-LABEL: {{^}}test_sink_global_small_min_scratch_global_neg1_offset:
define amdgpu_kernel void @test_sink_global_small_min_scratch_global_neg1_offset(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %out.gep = getelementptr i32, ptr addrspace(1) %out, i64 99999
  %in.gep = getelementptr i8, ptr addrspace(1) %in, i64 -4097
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i8, ptr addrspace(1) %in.gep
  %tmp2 = sext i8 %tmp1 to i32
  br label %endif

endif:
  %x = phi i32 [ %tmp2, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(1) %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_sink_small_offset_ds_append(
; OPT: %sunkaddr = getelementptr i8, ptr addrspace(3) %in, i32 28
; OPT: %tmp1 = call i32 @llvm.amdgcn.ds.append.p3(ptr addrspace(3) %sunkaddr, i1 false)
define amdgpu_kernel void @test_sink_small_offset_ds_append(ptr addrspace(3) %out, ptr addrspace(3) %in) {
entry:
  %out.gep = getelementptr i32, ptr addrspace(3) %out, i32 999999
  %in.gep = getelementptr i32, ptr addrspace(3) %in, i32 7
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = call i32 @llvm.amdgcn.ds.append.p3(ptr addrspace(3) %in.gep, i1 false)
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(3) %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_sink_small_offset_ds_consume(
; OPT: %sunkaddr = getelementptr i8, ptr addrspace(3) %in, i32 28
; OPT: %tmp1 = call i32 @llvm.amdgcn.ds.consume.p3(ptr addrspace(3) %sunkaddr, i1 false)
define amdgpu_kernel void @test_sink_small_offset_ds_consume(ptr addrspace(3) %out, ptr addrspace(3) %in) {
entry:
  %out.gep = getelementptr i32, ptr addrspace(3) %out, i32 999999
  %in.gep = getelementptr i32, ptr addrspace(3) %in, i32 7
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = call i32 @llvm.amdgcn.ds.consume.p3(ptr addrspace(3) %in.gep, i1 false)
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, ptr addrspace(3) %out.gep
  br label %done

done:
  ret void
}

declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #0
declare i32 @llvm.amdgcn.ds.append.p3(ptr addrspace(3) nocapture, i1 immarg) #3
declare i32 @llvm.amdgcn.ds.consume.p3(ptr addrspace(3) nocapture, i1 immarg) #3

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
attributes #2 = { nounwind argmemonly }
attributes #3 = { argmemonly convergent nounwind willreturn }
