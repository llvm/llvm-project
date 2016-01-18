; RUN: opt -S -codegenprepare -mtriple=amdgcn-unknown-unknown -mcpu=bonaire < %s | FileCheck -check-prefix=OPT -check-prefix=OPT-CI %s
; RUN: opt -S -codegenprepare -mtriple=amdgcn-unknown-unknown -mcpu=tonga < %s | FileCheck -check-prefix=OPT -check-prefix=OPT-VI %s
; RUN: llc -march=amdgcn -mcpu=bonaire -mattr=-promote-alloca < %s | FileCheck -check-prefix=GCN -check-prefix=CI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-promote-alloca < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; OPT-LABEL: @test_no_sink_flat_small_offset_i32(
; OPT: getelementptr i32, i32 addrspace(4)* %in
; OPT: br i1
; OPT-NOT: ptrtoint

; GCN-LABEL: {{^}}test_no_sink_flat_small_offset_i32:
; GCN: flat_load_dword
; GCN: {{^}}BB0_2:
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
