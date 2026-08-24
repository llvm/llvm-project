; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 < %s | FileCheck %s --check-prefix=GCN
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 -global-isel -global-isel-abort=1 < %s | FileCheck %s --check-prefix=GCN
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -O2 < %s | FileCheck %s --check-prefix=GCN
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -O2 -global-isel -global-isel-abort=1 < %s | FileCheck %s --check-prefix=GCN
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 < %s | FileCheck %s --check-prefix=GCN
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 -global-isel -global-isel-abort=1 < %s | FileCheck %s --check-prefix=GCN
; At -O0 the DAG combiner never rewrites the inverted condition into a setcc, so
; these runs cover the bare-xor form of the query reaching branch lowering.
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O0 < %s | FileCheck %s --check-prefix=GCN
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O0 -global-isel -global-isel-abort=1 < %s | FileCheck %s --check-prefix=GCN
; RUN: opt -passes=lower-expect -S %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 | FileCheck %s --check-prefix=EXPECT
; RUN: opt -passes=lower-expect -S %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 -global-isel -global-isel-abort=1 | FileCheck %s --check-prefix=EXPECT
; RUN: opt -passes=lower-expect -S %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 -stop-after=finalize-isel | FileCheck %s --check-prefix=MIR
; RUN: opt -passes=lower-expect -S %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 -global-isel -global-isel-abort=1 -stop-after=finalize-isel | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 < %s | FileCheck %s --check-prefix=LOC
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 -global-isel -global-isel-abort=1 < %s | FileCheck %s --check-prefix=LOC

declare noundef i1 @llvm.is.debugging.enabled()
declare i1 @llvm.expect.i1(i1, i1 immarg)
declare void @llvm.debugtrap()
declare i32 @llvm.amdgcn.ballot.i32(i1)
declare i32 @llvm.amdgcn.workgroup.id.x()
declare i32 @llvm.amdgcn.workitem.id.x()

; GCN-NOT: s_getreg_b32
; EXPECT-NOT: s_getreg_b32
; LOC-NOT: s_getreg_b32

define amdgpu_kernel void @direct_true_first(ptr addrspace(1) %out) {
entry:
  %enabled = call i1 @llvm.is.debugging.enabled()
  br i1 %enabled, label %debug, label %normal

debug:
  store volatile i32 1, ptr addrspace(1) %out
  br label %normal

normal:
  ret void
}

; GCN-LABEL: direct_true_first:
; GCN: s_cbranch_cdbgsys_or_user [[TRUE_FIRST_DEBUG:.LBB[0-9_]+]]
; GCN-NOT: s_cbranch_cdbgsys_or_user
; GCN-NOT: s_cbranch_scc
; GCN-NOT: s_cbranch_vcc
; GCN-NOT: s_cmp
; GCN: [[TRUE_FIRST_DEBUG]]:
; GCN: global_store_b32

define amdgpu_kernel void @direct_false_first(ptr addrspace(1) %out) {
entry:
  %enabled = call i1 @llvm.is.debugging.enabled()
  br i1 %enabled, label %debug, label %normal

normal:
  ret void

debug:
  store volatile i32 1, ptr addrspace(1) %out
  br label %normal
}

; GCN-LABEL: direct_false_first:
; GCN: s_cbranch_cdbgsys_or_user [[FALSE_FIRST_DEBUG:.LBB[0-9_]+]]
; GCN-NOT: s_cbranch_cdbgsys_or_user
; GCN-NOT: s_cbranch_scc
; GCN-NOT: s_cbranch_vcc
; GCN-NOT: s_cmp
; GCN: [[FALSE_FIRST_DEBUG]]:
; GCN: global_store_b32

define amdgpu_kernel void @chain_ordering(ptr addrspace(1) %out) {
entry:
  store volatile i32 0, ptr addrspace(1) %out
  %enabled = call i1 @llvm.is.debugging.enabled()
  br i1 %enabled, label %debug, label %normal

debug:
  store volatile i32 1, ptr addrspace(1) %out
  br label %normal

normal:
  ret void
}

; GCN-LABEL: chain_ordering:
; GCN: global_store_b32
; GCN: s_cbranch_cdbgsys_or_user [[CHAIN_DEBUG:.LBB[0-9_]+]]
; GCN-NOT: s_cbranch_scc
; GCN-NOT: s_cbranch_vcc
; GCN: [[CHAIN_DEBUG]]:
; GCN: global_store_b32

define amdgpu_kernel void @expected_debug_break() {
entry:
  %raw = call i1 @llvm.is.debugging.enabled()
  %enabled = call i1 @llvm.expect.i1(i1 %raw, i1 false)
  br i1 %enabled, label %debug, label %normal

debug:
  call void @llvm.debugtrap()
  br label %normal

normal:
  ret void
}

; GCN-LABEL: expected_debug_break:
; GCN-NOT: s_cbranch_execz
; GCN: s_cbranch_cdbgsys_or_user [[EXPECTED_DEBUG:.LBB[0-9_]+]]
; GCN-NOT: s_cbranch_cdbgsys_or_user
; GCN-NOT: s_cbranch_scc
; GCN-NOT: s_cbranch_vcc
; GCN-NOT: s_cmp
; GCN: [[EXPECTED_DEBUG]]:
; GCN-NEXT: s_trap 3

; EXPECT-LABEL: expected_debug_break:
; EXPECT: s_cbranch_cdbgsys_or_user [[COLD_DEBUG:.LBB[0-9_]+]]
; EXPECT-NOT: s_branch
; EXPECT: ; %bb.1:{{.*}}%normal
; EXPECT: s_endpgm
; EXPECT: [[COLD_DEBUG]]:
; EXPECT-COUNT-1: s_trap 3

; MIR-LABEL: name: expected_debug_break
; MIR: bb.{{[0-9]+}}.entry:
; MIR: successors: %bb.[[MIR_DEBUG:[0-9]+]](0x00106035), %bb.[[MIR_NORMAL:[0-9]+]](0x7fef9fcb)
; MIR: nomerge S_CBRANCH_CDBGSYS_OR_USER %bb.[[MIR_DEBUG]]
; MIR-NEXT: S_BRANCH %bb.[[MIR_NORMAL]]

define amdgpu_kernel void @arithmetic_between(i32 %x,
                                               ptr addrspace(1) %out) {
entry:
  %enabled = call i1 @llvm.is.debugging.enabled()
  %value = add i32 %x, 1
  br i1 %enabled, label %debug, label %exit

debug:
  store volatile i32 1, ptr addrspace(1) %out
  br label %exit

exit:
  store volatile i32 %value, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: arithmetic_between:
; GCN-NOT: s_getreg_b32
; GCN: s_cbranch_cdbgsys_or_user

; Divergence alone does not block fusion.
define amdgpu_kernel void @divergent_value_between(ptr addrspace(1) %out) {
entry:
  %enabled = call i1 @llvm.is.debugging.enabled()
  %value = call i32 @llvm.amdgcn.workitem.id.x()
  br i1 %enabled, label %debug, label %exit

debug:
  store volatile i32 1, ptr addrspace(1) %out
  br label %exit

exit:
  store volatile i32 %value, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: divergent_value_between:
; GCN-NOT: s_getreg_b32
; GCN: s_cbranch_cdbgsys_or_user

; Convergent without memory or side effects does not block fusion.
define amdgpu_kernel void @ballot_between(ptr addrspace(1) %out) {
entry:
  %enabled = call i1 @llvm.is.debugging.enabled()
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %predicate = icmp eq i32 %id, 0
  %value = call i32 @llvm.amdgcn.ballot.i32(i1 %predicate)
  br i1 %enabled, label %debug, label %exit

debug:
  store volatile i32 1, ptr addrspace(1) %out
  br label %exit

exit:
  store volatile i32 %value, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: ballot_between:
; GCN-NOT: s_getreg_b32
; GCN: s_cbranch_cdbgsys_or_user

define amdgpu_kernel void @debug_location(ptr addrspace(1) %out) !dbg !4 {
entry:
  %enabled = call i1 @llvm.is.debugging.enabled(), !dbg !7
  br i1 %enabled, label %debug, label %normal, !dbg !8

debug:
  store volatile i32 1, ptr addrspace(1) %out, !dbg !9
  br label %normal, !dbg !10

normal:
  ret void, !dbg !11
}

; LOC-LABEL: debug_location:
; LOC: .loc {{[0-9]+}} 3 3
; LOC-NEXT: s_cbranch_cdbgsys_or_user

define amdgpu_kernel void @uniform_guard_debugtrap(ptr addrspace(1) %out) {
entry:
  %wg = call i32 @llvm.amdgcn.workgroup.id.x()
  %is3 = icmp eq i32 %wg, 3
  br i1 %is3, label %region, label %exit

region:
  %e = call i1 @llvm.is.debugging.enabled()
  br i1 %e, label %dbg, label %after

dbg:
  call void @llvm.debugtrap()
  br label %after

after:
  store volatile i32 7, ptr addrspace(1) %out
  br label %exit

exit:
  ret void
}

; GCN-LABEL: uniform_guard_debugtrap:
; GCN-NOT: execz
; GCN-NOT: v_cmpx
; GCN: s_cbranch_{{scc1|vccnz}} [[UNIFORM_SKIP:.LBB[0-9_]+]]
; GCN: s_cbranch_cdbgsys_or_user [[UNIFORM_DEBUG:.LBB[0-9_]+]]
; GCN: [[UNIFORM_DEBUG]]:
; GCN-NEXT: s_trap 3
; GCN: [[UNIFORM_SKIP]]:

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "llvm", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "is-debugging-enabled.ll", directory: "/")
!2 = !{i32 2, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "debug_location", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 2, column: 3, scope: !4)
!8 = !DILocation(line: 3, column: 3, scope: !4)
!9 = !DILocation(line: 4, column: 3, scope: !4)
!10 = !DILocation(line: 5, column: 3, scope: !4)
!11 = !DILocation(line: 6, column: 3, scope: !4)
