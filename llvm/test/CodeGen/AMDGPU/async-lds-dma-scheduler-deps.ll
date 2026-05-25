; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -debug-only=machine-scheduler < %s 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts

@lds1 = internal addrspace(3) global i32 poison, align 4
@lds2 = internal addrspace(3) global i32 poison, align 4

declare void @llvm.amdgcn.raw.buffer.load.async.lds(<4 x i32>, ptr addrspace(3), i32, i32, i32, i32, i32)
declare void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32>, ptr addrspace(3), i32, i32, i32, i32, i32)
declare void @llvm.amdgcn.asyncmark()
declare void @llvm.amdgcn.wait.asyncmark(i16)
declare i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32>, i32, i32, i32)
declare void @llvm.amdgcn.raw.buffer.store.i32(i32, <4 x i32>, i32, i32, i32)

; CHECK-LABEL: async_load_with_asyncmark:%bb.0
; CHECK-LABEL: SU(6): BUFFER_LOAD_DWORD_LDS_OFFSET_ASYNC{{.*}}implicit $asynccnt
; CHECK-LABEL:   Successors:
; CHECK-NEXT:     SU(7): Anti Latency=0
; CHECK-NEXT:     ExitSU: Ord  Latency={{[0-9]+}} Artificial
; CHECK-NEXT:     SU(8): Ord  Latency=0 Memory
; CHECK-NEXT:     Pressure Diff
; CHECK-LABEL: SU(7): ASYNCMARK
; CHECK-LABEL:   Predecessors:
; CHECK-NEXT:     SU(6): Anti Latency=0
; CHECK-NEXT:   Successors:
; CHECK-NEXT:     SU(8): Data Latency=1 Reg=$asynccnt
; CHECK-NEXT:     Pressure Diff
; CHECK-LABEL: SU(8): BUFFER_LOAD_DWORD_LDS_OFFSET_ASYNC{{.*}}implicit $asynccnt
define amdgpu_ps void @async_load_with_asyncmark(<4 x i32> inreg %rsrc, ptr addrspace(3) inreg %lds) {
  call void @llvm.amdgcn.raw.buffer.load.async.lds(<4 x i32> %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.raw.buffer.load.async.lds(<4 x i32> %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 0, i32 0)
  ret void
}

; CHECK-LABEL: nonasync_load_with_asyncmark:%bb.0
; CHECK-LABEL: SU(6): ASYNCMARK
; CHECK-NEXT:   # preds left{{.*}}: 0
; CHECK-NEXT:   # succs left{{.*}}: 0
define amdgpu_ps void @nonasync_load_with_asyncmark(<4 x i32> inreg %rsrc) {
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds1, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds2, i32 4, i32 0, i32 0, i32 0, i32 0)
  ret void
}

; CHECK-LABEL: two_batches:%bb.0
;
; CHECK-LABEL: SU(6): BUFFER_LOAD_DWORD_LDS_OFFSET_ASYNC{{.*}}implicit $asynccnt
; CHECK-LABEL:   Predecessors:
; CHECK-LABEL:   Successors:
; CHECK:          SU(8): Anti Latency=0
; CHECK-LABEL: Pressure Diff
;
; CHECK-LABEL: SU(7): BUFFER_LOAD_DWORD_LDS_OFFSET_ASYNC{{.*}}implicit $asynccnt
; CHECK-LABEL:   Predecessors:
; CHECK-LABEL:   Successors:
; CHECK:          SU(8): Anti Latency=0
; CHECK-LABEL: Pressure Diff
;
; CHECK-LABEL: SU(8): ASYNCMARK
; CHECK-LABEL:   Predecessors:
; CHECK:          SU(7): Anti Latency=0
; CHECK:          SU(6): Anti Latency=0
; CHECK-NOT:      SU(6): Ord  Latency=0 Barrier
; CHECK-NOT:      SU(7): Ord  Latency=0 Barrier
; CHECK-LABEL:   Successors:
; CHECK:          SU(10): Data Latency=1 Reg=$asynccnt
; CHECK:          SU(9): Data Latency=1 Reg=$asynccnt
; CHECK:          SU(11): Anti Latency=0
; CHECK:          SU(10): Anti Latency=0
; CHECK-NOT:      SU(9): Ord  Latency=0 Barrier
; CHECK-NOT:      Reg=%
; CHECK-NOT:      Reg=$m0
; CHECK-LABEL: Pressure Diff
;
; CHECK-LABEL: SU(9): BUFFER_LOAD_DWORD_LDS_OFFSET_ASYNC{{.*}}implicit $asynccnt
; CHECK-LABEL:   Predecessors:
; CHECK:          SU(8): Data Latency=1 Reg=$asynccnt
; CHECK-LABEL:   Successors:
; CHECK:          SU(11): Anti Latency=0
; CHECK:          SU(10): Anti Latency=0
; CHECK-LABEL: Pressure Diff
;
; CHECK-LABEL: SU(10): ASYNCMARK
; CHECK-LABEL:   Predecessors:
; CHECK:          SU(9): Anti Latency=0
; CHECK:          SU(8): Data Latency=1 Reg=$asynccnt
; CHECK:          SU(8): Anti Latency=0
; CHECK:          SU(7): Anti Latency=0
; CHECK:          SU(6): Anti Latency=0
; CHECK-NOT:      SU(9): Ord  Latency=0 Barrier
; CHECK-LABEL:   Successors:
; CHECK:          SU(11): Data Latency=1 Reg=$asynccnt
; CHECK:          SU(11): Anti Latency=0
; CHECK-NOT:      SU(11): Ord  Latency=0 Barrier
; CHECK-NOT:      Reg=%
; CHECK-NOT:      Reg=$m0
; CHECK-LABEL: Pressure Diff
;
; CHECK-LABEL: SU(11): WAIT_ASYNCMARK
; CHECK-LABEL:   Predecessors:
; CHECK:          SU(10): Data Latency=1 Reg=$asynccnt
; CHECK:          SU(10): Anti Latency=0
; CHECK:          SU(9): Anti Latency=0
; CHECK:          SU(9): Ord  Latency=0 Barrier
; CHECK:          SU(8): Anti Latency=0
; CHECK:          SU(7): Anti Latency=0
; CHECK:          SU(7): Ord  Latency=0 Barrier
; CHECK:          SU(6): Anti Latency=0
; CHECK:          SU(6): Ord  Latency=0 Barrier
; CHECK-NOT:      Reg=%
; CHECK-NOT:      Reg=$m0
; CHECK-LABEL: Pressure Diff
define amdgpu_ps void @two_batches(<4 x i32> inreg %rsrc, ptr addrspace(3) inreg %lds) {
  call void @llvm.amdgcn.raw.buffer.load.async.lds(<4 x i32> %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.async.lds(<4 x i32> %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.raw.buffer.load.async.lds(<4 x i32> %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.wait.asyncmark(i16 0)
  ret void
}

; CHECK-LABEL: mixed_around_mark:%bb.0
;
; CHECK-LABEL: SU(9): GLOBAL_STORE_DWORD
; CHECK-LABEL:   Predecessors:
; CHECK-LABEL:   Successors:
; CHECK-NOT:      SU(13): ASYNCMARK
; CHECK:          SU(17): Ord  Latency=0 Barrier
; CHECK-NOT:      Reg=$asynccnt
; CHECK-NOT:      Reg=$asynccnt
; CHECK-LABEL: Pressure Diff
;
; CHECK-LABEL: SU(12): BUFFER_LOAD_DWORD_LDS_OFFSET_ASYNC{{.*}}implicit $asynccnt
; CHECK-LABEL:   Predecessors:
; CHECK-LABEL:   Successors:
; CHECK:          SU(17): Anti Latency=0
; CHECK:          SU(13): Anti Latency=0
; CHECK:          SU(17): Ord  Latency=0 Barrier
; CHECK-NOT:      Reg=$asynccnt
; CHECK-LABEL: Pressure Diff
;
; CHECK-LABEL: SU(13): ASYNCMARK
; CHECK-LABEL:   Predecessors:
; CHECK-NOT:      Ord
; CHECK-NOT:      SU(9):
; CHECK-NOT:      Reg=%
; CHECK-NOT:      Reg=$m0
; CHECK:          SU(12): Anti Latency=0
; CHECK-LABEL:   Successors:
; CHECK-NOT:      Ord
; CHECK-NOT:      SU(14):
; CHECK-NOT:      SU(15):
; CHECK-NOT:      Reg=%
; CHECK-NOT:      Reg=$m0
; CHECK-LABEL: Pressure Diff
;
; CHECK-LABEL: SU(14): DS_WRITE_B32_gfx9
; CHECK-LABEL:   Predecessors:
; CHECK-NOT:      SU(13):
; CHECK-LABEL:   Successors:
; CHECK:          SU(17): Ord  Latency=0 Barrier
; CHECK-NOT:      Reg=$asynccnt
; CHECK-NOT:      Reg=$asynccnt
; CHECK-LABEL: Pressure Diff
;
; CHECK-LABEL: SU(15): BUFFER_STORE_DWORD_OFFSET_exact
; CHECK-LABEL:   Predecessors:
; CHECK-NOT:      SU(13):
; CHECK-LABEL:   Successors:
; CHECK:          SU(17): Ord  Latency=0 Barrier
; CHECK-NOT:      Reg=$asynccnt
; CHECK-NOT:      Reg=$asynccnt
; CHECK-LABEL: Pressure Diff
;
; CHECK-LABEL: SU(16): BUFFER_LOAD_DWORD_LDS_OFFSET_ASYNC{{.*}}implicit $asynccnt
; CHECK-LABEL:   Predecessors:
; CHECK:          SU(13): Data Latency=1 Reg=$asynccnt
; CHECK-LABEL:   Successors:
; CHECK:          SU(17): Anti Latency=0
; CHECK:          SU(17): Ord  Latency=0 Barrier
; CHECK-NOT:      Reg=$asynccnt
; CHECK-LABEL: Pressure Diff
;
; CHECK-LABEL: SU(17): WAIT_ASYNCMARK
; CHECK-LABEL:   Predecessors:
; CHECK:          SU(16): Anti Latency=0
; CHECK:          SU(16): Ord  Latency=0 Barrier
; CHECK:          SU(15): Ord  Latency=0 Barrier
; CHECK:          SU(14): Ord  Latency=0 Barrier
; CHECK:          SU(13): Data Latency=1 Reg=$asynccnt
; CHECK:          SU(13): Anti Latency=0
; CHECK:          SU(12): Anti Latency=0
; CHECK:          SU(12): Ord  Latency=0 Barrier
; CHECK:          SU(9): Ord  Latency=0 Barrier
; CHECK-NOT:      Reg=%
; CHECK-NOT:      Reg=$m0
; CHECK-LABEL: Pressure Diff
define amdgpu_ps void @mixed_around_mark(<4 x i32> inreg %rsrc, ptr addrspace(1) %g, ptr addrspace(3) %l, ptr addrspace(3) %l2, i32 %x) {
  %a = load i32, ptr addrspace(1) %g
  %b = load i32, ptr addrspace(3) %l2
  store i32 %x, ptr addrspace(1) %g
  call void @llvm.amdgcn.raw.buffer.load.async.lds(<4 x i32> %rsrc, ptr addrspace(3) %l, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  %c = call i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32> %rsrc, i32 0, i32 0, i32 0)
  store i32 %x, ptr addrspace(3) %l2
  call void @llvm.amdgcn.raw.buffer.store.i32(i32 %x, <4 x i32> %rsrc, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.async.lds(<4 x i32> %rsrc, ptr addrspace(3) %l, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.wait.asyncmark(i16 0)
  ret void
}
