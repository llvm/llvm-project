; RUN: opt -S -codegenprepare -mtriple=amdgcn-unknown-unknown -mcpu=bonaire < %s | FileCheck --check-prefixes=OPT,OPT-CI,OPT-CIVI %s
; RUN: opt -S -codegenprepare -mtriple=amdgcn-unknown-unknown -mcpu=tonga -mattr=-flat-for-global < %s | FileCheck --check-prefixes=OPT,OPT-CIVI %s
; RUN: opt -S -codegenprepare -mtriple=amdgcn-unknown-unknown -mcpu=gfx900 -mattr=-flat-for-global < %s | FileCheck --check-prefixes=OPT,OPT-GFX9 %s
; RUN: llc -march=amdgcn -amdgpu-scalarize-global-loads=false -mcpu=bonaire -mattr=-promote-alloca < %s | FileCheck --check-prefixes=GCN,CI,CIVI %s
; RUN: llc -march=amdgcn -amdgpu-scalarize-global-loads=false -mcpu=tonga -mattr=-flat-for-global -mattr=-promote-alloca < %s | FileCheck --check-prefixes=GCN,CIVI %s
; RUN: llc -march=amdgcn -amdgpu-scalarize-global-loads=false -mcpu=gfx900 -mattr=-flat-for-global -mattr=-promote-alloca < %s | FileCheck --check-prefixes=GCN,GFX9 %s

; OPT-LABEL: @test_no_sink_flat_small_offset_i32(
; OPT-CIVI: getelementptr i32, i32* %in
; OPT-CIVI: br i1
; OPT-CIVI-NOT: ptrtoint

; OPT-GFX9: br
; OPT-GFX9: %sunkaddr = getelementptr i8, i8* %0, i64 28
; OPT-GFX9: %1 = bitcast i8* %sunkaddr to i32*
; OPT-GFX9: load i32, i32* %1

; GCN-LABEL: {{^}}test_no_sink_flat_small_offset_i32:
; GCN: flat_load_dword
; GCN: {{^}}.LBB0_2:
define amdgpu_kernel void @test_no_sink_flat_small_offset_i32(i32* %out, i32* %in, i32 %cond) {
entry:
  %out.gep = getelementptr i32, i32* %out, i64 999999
  %in.gep = getelementptr i32, i32* %in, i64 7
  %tmp0 = icmp eq i32 %cond, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i32, i32* %in.gep
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, i32* %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_sink_noop_addrspacecast_flat_to_global_i32(
; OPT: getelementptr i32, i32* %out,
; rOPT-CI-NOT: getelementptr
; OPT: br i1

; OPT-CI: addrspacecast
; OPT-CI: getelementptr
; OPT-CI: bitcast
; OPT: br label

; GCN-LABEL: {{^}}test_sink_noop_addrspacecast_flat_to_global_i32:
; CI: buffer_load_dword {{v[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:28
define amdgpu_kernel void @test_sink_noop_addrspacecast_flat_to_global_i32(i32* %out, i32* %in, i32 %cond) {
entry:
  %out.gep = getelementptr i32, i32* %out, i64 999999
  %in.gep = getelementptr i32, i32* %in, i64 7
  %cast = addrspacecast i32* %in.gep to i32 addrspace(1)*
  %tmp0 = icmp eq i32 %cond, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i32, i32 addrspace(1)* %cast
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, i32* %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_sink_noop_addrspacecast_flat_to_constant_i32(
; OPT: getelementptr i32, i32* %out,
; OPT-CI-NOT: getelementptr
; OPT: br i1

; OPT-CI: addrspacecast
; OPT-CI: getelementptr
; OPT-CI: bitcast
; OPT: br label

; GCN-LABEL: {{^}}test_sink_noop_addrspacecast_flat_to_constant_i32:
; CI: s_load_dword {{s[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0xd
define amdgpu_kernel void @test_sink_noop_addrspacecast_flat_to_constant_i32(i32* %out, i32* %in, i32 %cond) {
entry:
  %out.gep = getelementptr i32, i32* %out, i64 999999
  %in.gep = getelementptr i32, i32* %in, i64 7
  %cast = addrspacecast i32* %in.gep to i32 addrspace(4)*
  %tmp0 = icmp eq i32 %cond, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i32, i32 addrspace(4)* %cast
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, i32* %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_sink_flat_small_max_flat_offset(
; OPT-CIVI: %in.gep = getelementptr i8, i8* %in, i64 4095
; OPT-CIVI: br
; OPT-CIVI-NOT: getelementptr
; OPT-CIVI: load i8, i8* %in.gep

; OPT-GFX9: br
; OPT-GFX9: %sunkaddr = getelementptr i8, i8* %in, i64 4095
; OPT-GFX9: load i8, i8* %sunkaddr

; GCN-LABEL: {{^}}test_sink_flat_small_max_flat_offset:
; GFX9: flat_load_sbyte v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}} offset:4095{{$}}
; CIVI: flat_load_sbyte v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @test_sink_flat_small_max_flat_offset(i32* %out, i8* %in) #1 {
entry:
  %out.gep = getelementptr i32, i32* %out, i32 1024
  %in.gep = getelementptr i8, i8* %in, i64 4095
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i8, i8* %in.gep
  %tmp2 = sext i8 %tmp1 to i32
  br label %endif

endif:
  %x = phi i32 [ %tmp2, %if ], [ 0, %entry ]
  store i32 %x, i32* %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_sink_flat_small_max_plus_1_flat_offset(
; OPT: %in.gep = getelementptr i8, i8* %in, i64 4096
; OPT: br
; OPT-NOT: getelementptr
; OPT: load i8, i8* %in.gep

; GCN-LABEL: {{^}}test_sink_flat_small_max_plus_1_flat_offset:
; GCN: flat_load_sbyte v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @test_sink_flat_small_max_plus_1_flat_offset(i32* %out, i8* %in) #1 {
entry:
  %out.gep = getelementptr i32, i32* %out, i64 99999
  %in.gep = getelementptr i8, i8* %in, i64 4096
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i8, i8* %in.gep
  %tmp2 = sext i8 %tmp1 to i32
  br label %endif

endif:
  %x = phi i32 [ %tmp2, %if ], [ 0, %entry ]
  store i32 %x, i32* %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_no_sink_flat_reg_offset(
; OPT: %in.gep = getelementptr i8, i8* %in, i64 %reg
; OPT: br

; OPT-NOT: getelementptr
; OPT: load i8, i8* %in.gep

; GCN-LABEL: {{^}}test_no_sink_flat_reg_offset:
; GCN: flat_load_sbyte v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @test_no_sink_flat_reg_offset(i32* %out, i8* %in, i64 %reg) #1 {
entry:
  %out.gep = getelementptr i32, i32* %out, i32 1024
  %in.gep = getelementptr i8, i8* %in, i64 %reg
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %tmp0 = icmp eq i32 %tid, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i8, i8* %in.gep
  %tmp2 = sext i8 %tmp1 to i32
  br label %endif

endif:
  %x = phi i32 [ %tmp2, %if ], [ 0, %entry ]
  store i32 %x, i32* %out.gep
  br label %done

done:
  ret void
}

declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
attributes #2 = { nounwind argmemonly }
