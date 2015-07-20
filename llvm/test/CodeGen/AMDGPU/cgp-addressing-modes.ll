; RUN: opt -S -codegenprepare -mtriple=amdgcn-unknown-unknown -mcpu=bonaire < %s | FileCheck -check-prefix=OPT -check-prefix=OPT-CI %s
; RUN: opt -S -codegenprepare -mtriple=amdgcn-unknown-unknown -mcpu=tonga < %s | FileCheck -check-prefix=OPT -check-prefix=OPT-VI %s
; RUN: llc -march=amdgcn -mcpu=bonaire -mattr=-promote-alloca < %s | FileCheck -check-prefix=GCN -check-prefix=CI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-promote-alloca < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

declare i32 @llvm.r600.read.tidig.x() #0

; OPT-LABEL: @test_sink_global_small_offset_i32(
; OPT-CI-NOT: getelementptr i32, i32 addrspace(1)* %in
; OPT-VI: getelementptr i32, i32 addrspace(1)* %in
; OPT: br i1
; OPT-CI: ptrtoint

; GCN-LABEL: {{^}}test_sink_global_small_offset_i32:
; GCN: {{^}}BB0_2:
define void @test_sink_global_small_offset_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in, i32 %cond) {
entry:
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i64 999999
  %in.gep = getelementptr i32, i32 addrspace(1)* %in, i64 7
  %tmp0 = icmp eq i32 %cond, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i32, i32 addrspace(1)* %in.gep
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, i32 addrspace(1)* %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_sink_global_small_max_i32_ds_offset(
; OPT: %in.gep = getelementptr i8, i8 addrspace(1)* %in, i64 65535
; OPT: br i1

; GCN-LABEL: {{^}}test_sink_global_small_max_i32_ds_offset:
; GCN: s_and_saveexec_b64
; GCN: buffer_load_sbyte {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, s{{[0-9]+$}}
; GCN: {{^}}BB1_2:
; GCN: s_or_b64 exec
define void @test_sink_global_small_max_i32_ds_offset(i32 addrspace(1)* %out, i8 addrspace(1)* %in, i32 %cond) {
entry:
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i64 99999
  %in.gep = getelementptr i8, i8 addrspace(1)* %in, i64 65535
  %tmp0 = icmp eq i32 %cond, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i8, i8 addrspace(1)* %in.gep
  %tmp2 = sext i8 %tmp1 to i32
  br label %endif

endif:
  %x = phi i32 [ %tmp2, %if ], [ 0, %entry ]
  store i32 %x, i32 addrspace(1)* %out.gep
  br label %done

done:
  ret void
}

; GCN-LABEL: {{^}}test_sink_global_small_max_mubuf_offset:
; GCN: s_and_saveexec_b64
; GCN: buffer_load_sbyte {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offset:4095{{$}}
; GCN: {{^}}BB2_2:
; GCN: s_or_b64 exec
define void @test_sink_global_small_max_mubuf_offset(i32 addrspace(1)* %out, i8 addrspace(1)* %in, i32 %cond) {
entry:
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i32 1024
  %in.gep = getelementptr i8, i8 addrspace(1)* %in, i64 4095
  %tmp0 = icmp eq i32 %cond, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i8, i8 addrspace(1)* %in.gep
  %tmp2 = sext i8 %tmp1 to i32
  br label %endif

endif:
  %x = phi i32 [ %tmp2, %if ], [ 0, %entry ]
  store i32 %x, i32 addrspace(1)* %out.gep
  br label %done

done:
  ret void
}

; GCN-LABEL: {{^}}test_sink_global_small_max_plus_1_mubuf_offset:
; GCN: s_and_saveexec_b64
; GCN: buffer_load_sbyte {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, s{{[0-9]+$}}
; GCN: {{^}}BB3_2:
; GCN: s_or_b64 exec
define void @test_sink_global_small_max_plus_1_mubuf_offset(i32 addrspace(1)* %out, i8 addrspace(1)* %in, i32 %cond) {
entry:
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i64 99999
  %in.gep = getelementptr i8, i8 addrspace(1)* %in, i64 4096
  %tmp0 = icmp eq i32 %cond, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i8, i8 addrspace(1)* %in.gep
  %tmp2 = sext i8 %tmp1 to i32
  br label %endif

endif:
  %x = phi i32 [ %tmp2, %if ], [ 0, %entry ]
  store i32 %x, i32 addrspace(1)* %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_no_sink_flat_small_offset_i32(
; OPT: getelementptr i32, i32 addrspace(4)* %in
; OPT: br i1
; OPT-NOT: ptrtoint

; GCN-LABEL: {{^}}test_no_sink_flat_small_offset_i32:
; GCN: flat_load_dword
; GCN: {{^}}BB4_2:

define void @test_no_sink_flat_small_offset_i32(i32 addrspace(4)* %out, i32 addrspace(4)* %in, i32 %cond) {
entry:
  %out.gep = getelementptr i32, i32 addrspace(4)* %out, i64 999999
  %in.gep = getelementptr i32, i32 addrspace(4)* %in, i64 7
  %tmp0 = icmp eq i32 %cond, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i32, i32 addrspace(4)* %in.gep
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, i32 addrspace(4)* %out.gep
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_sink_scratch_small_offset_i32(
; OPT-NOT:  getelementptr [512 x i32]
; OPT: br i1
; OPT: ptrtoint

; GCN-LABEL: {{^}}test_sink_scratch_small_offset_i32:
; GCN: s_and_saveexec_b64
; GCN: buffer_store_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, {{s[0-9]+}} offen offset:4092{{$}}
; GCN: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, {{s[0-9]+}} offen offset:4092{{$}}
; GCN: {{^}}BB5_2:
define void @test_sink_scratch_small_offset_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in, i32 %cond, i32 %arg) {
entry:
  %alloca = alloca [512 x i32], align 4
  %out.gep.0 = getelementptr i32, i32 addrspace(1)* %out, i64 999998
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i64 999999
  %add.arg = add i32 %arg, 8
  %alloca.gep = getelementptr [512 x i32], [512 x i32]* %alloca, i32 0, i32 1023
  %tmp0 = icmp eq i32 %cond, 0
  br i1 %tmp0, label %endif, label %if

if:
  store volatile i32 123, i32* %alloca.gep
  %tmp1 = load volatile i32, i32* %alloca.gep
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, i32 addrspace(1)* %out.gep.0
  %load = load volatile i32, i32* %alloca.gep
  store i32 %load, i32 addrspace(1)* %out.gep.1
  br label %done

done:
  ret void
}

; OPT-LABEL: @test_no_sink_scratch_large_offset_i32(
; OPT: %alloca.gep = getelementptr [512 x i32], [512 x i32]* %alloca, i32 0, i32 1024
; OPT: br i1
; OPT-NOT: ptrtoint

; GCN-LABEL: {{^}}test_no_sink_scratch_large_offset_i32:
; GCN: s_and_saveexec_b64
; GCN: buffer_store_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, {{s[0-9]+}} offen{{$}}
; GCN: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, {{s[0-9]+}} offen{{$}}
; GCN: {{^}}BB6_2:
define void @test_no_sink_scratch_large_offset_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in, i32 %cond, i32 %arg) {
entry:
  %alloca = alloca [512 x i32], align 4
  %out.gep.0 = getelementptr i32, i32 addrspace(1)* %out, i64 999998
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i64 999999
  %add.arg = add i32 %arg, 8
  %alloca.gep = getelementptr [512 x i32], [512 x i32]* %alloca, i32 0, i32 1024
  %tmp0 = icmp eq i32 %cond, 0
  br i1 %tmp0, label %endif, label %if

if:
  store volatile i32 123, i32* %alloca.gep
  %tmp1 = load volatile i32, i32* %alloca.gep
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, i32 addrspace(1)* %out.gep.0
  %load = load volatile i32, i32* %alloca.gep
  store i32 %load, i32 addrspace(1)* %out.gep.1
  br label %done

done:
  ret void
}

; GCN-LABEL: {{^}}test_sink_global_vreg_sreg_i32:
; VI-DAG: s_movk_i32 flat_scratch_lo, 0x0
; VI-DAG: s_movk_i32 flat_scratch_hi, 0x0
; GCN: s_and_saveexec_b64
; CI: buffer_load_dword {{v[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; VI: flat_load_dword v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
; GCN: {{^}}BB7_2:
define void @test_sink_global_vreg_sreg_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in, i32 %offset, i32 %cond) {
entry:
  %offset.ext = zext i32 %offset to i64
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i64 999999
  %in.gep = getelementptr i32, i32 addrspace(1)* %in, i64 %offset.ext
  %tmp0 = icmp eq i32 %cond, 0
  br i1 %tmp0, label %endif, label %if

if:
  %tmp1 = load i32, i32 addrspace(1)* %in.gep
  br label %endif

endif:
  %x = phi i32 [ %tmp1, %if ], [ 0, %entry ]
  store i32 %x, i32 addrspace(1)* %out.gep
  br label %done

done:
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
