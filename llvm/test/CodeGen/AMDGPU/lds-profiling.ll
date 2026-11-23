; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK,ON %s
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -enable-new-pm -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK,ON %s
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -O0 -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK,ON-O0 %s
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -amdgpu-lds-profiling-clones=0 < %s | FileCheck -check-prefixes=CHECK,OFF %s

; The kernel is emitted with a profiling clone, which measures the bank
; conflict of each LDS access before making it. The LDS instructions of the
; kernel and of the clone are listed in a PC section with their site ids.

@lds = internal addrspace(3) global [256 x i32] poison, align 4

declare i32 @llvm.amdgcn.workitem.id.x()

; The kernel itself has no instrumentation.
; CHECK-LABEL: {{^}}k:
; CHECK-NOT: global_atomic_add
; CHECK-NOT: k.prof.counters
; OFF-NOT: .Lpcsection
; ON: [[ORIG0:.Lpcsection[0-9]+]]:
; ON-NEXT: ds_store_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; ON: [[ORIG1:.Lpcsection[0-9]+]]:
; ON-NEXT: ds_load_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:284
; CHECK: .amdhsa_kernel k
; CHECK: .amdhsa_group_segment_fixed_size 1024

; Number of sites and hash of their sequence.
; ON: .section .AMDGPU.ldsprof,"",@progbits
; ON: .type k.ldsprof,@object
; ON-NEXT: k.ldsprof:
; ON-NEXT: .long 2
; ON-NEXT: .long [[HASH:[0-9]+]]
; ON-NEXT: .size k.ldsprof, 8

; Address, site id and kind of each LDS instruction.
; ON: .section amdgpu_lds_prof_sites,"awo",@progbits,.text
; ON-NEXT: [[BASE0:.Lpcsection_base[0-9]+]]:
; ON-NEXT: .long [[ORIG0]]-[[BASE0]]
; ON-NEXT: .long 0
; ON-NEXT: .long 1
; ON-NEXT: [[BASE1:.Lpcsection_base[0-9]+]]:
; ON-NEXT: .long [[ORIG1]]-[[BASE1]]
; ON-NEXT: .long 1
; ON-NEXT: .long 1

; OFF-NOT: k.prof
; OFF-NOT: ldsprof
; OFF-NOT: amdgpu_lds_prof_sites

; ON-LABEL: {{^}}k.prof:
; No lane executes the access: not a sample.
; ON: s_mov_b32 [[EXEC:s[0-9]+]], exec_lo
; ON: s_cmp_eq_u32 [[EXEC]], 0
; d = (addr + offset) >> 2, limited to the 1024 byte allocation.
; ON: v_add_nc_u32_e32 [[SUM:v[0-9]+]], 16, [[ADDR:v[0-9]+]]
; ON: v_lshrrev_b32_e32 [[D:v[0-9]+]], 2, [[SUM]]
; ON: v_cmp_gt_u32_e64 [[ACTIVE:s[0-9]+]], s{{[0-9]+}}, [[D]]
; ON: v_and_b32_e32 [[BANK:v[0-9]+]], 63, [[D]]
; ON: s_cbranch_scc1 [[SKIP0:.LBB[0-9_]+]]
; Every lane is outside the allocation: a sample without a conflict.
; ON: s_cmp_eq_u32 [[ACTIVE]], 0
; ON: s_cbranch_scc1 [[PUBLISH0:.LBB[0-9_]+]]
; One representative lane per distinct dword.
; ON: [[LOOP1:.LBB[0-9_]+]]:
; ON: s_ctz_i32_b32 [[LANE:s[0-9]+]], [[ACTIVE]]
; ON: v_readlane_b32 [[VAL:s[0-9]+]], [[D]], [[LANE]]
; ON: v_cmp_eq_u32_e64 [[SAME:s[0-9]+]], [[VAL]], [[D]]
; ON: s_and_not1_b32 [[ACTIVE]], [[ACTIVE]], [[SAME]]
; ON: s_cbranch_scc1 [[LOOP1]]
; The largest number of representatives in one bank.
; ON: [[LOOP2:.LBB[0-9_]+]]:
; ON: v_readlane_b32 [[BVAL:s[0-9]+]], [[BANK]], s{{[0-9]+}}
; ON: v_cmp_eq_u32_e64 [[BSAME:s[0-9]+]], [[BVAL]], [[BANK]]
; ON: s_bcnt1_i32_b32 [[COUNT:s[0-9]+]], [[BSAME]]
; ON: s_max_u32 [[NWAY:s[0-9]+]], [[NWAY]], [[COUNT]]
; ON: s_cbranch_scc1 [[LOOP2]]
; One lane adds { n, 1, n, n != 0 } to the counters of site 0.
; ON-NEXT: [[PUBLISH0]]:
; ON-NEXT: s_min_u32 [[BASELINE:s[0-9]+]], [[NWAY]], s{{[0-9]+}}
; ON-NEXT: s_mov_b32 [[SAVE:s[0-9]+]], exec_lo
; ON-NEXT: s_mov_b32 exec_lo, 1
; ON-NEXT: v_dual_mov_b32 [[ZERO:v[0-9]+]], 0 :: v_dual_mov_b32 [[VN:v[0-9]+]], [[NWAY]]
; ON-NEXT: v_dual_mov_b32 [[ONE:v[0-9]+]], 1 :: v_dual_mov_b32 [[VBASE:v[0-9]+]], [[BASELINE]]
; ON-NEXT: s_get_pc_i64 [[PTR:s\[[0-9:]+\]]]
; ON-NEXT: s_add_nc_u64 [[PTR]], [[PTR]], k.prof.counters@rel64+4
; ON-NEXT: s_clause 0x3
; ON-NEXT: global_atomic_add_u32 [[ZERO]], [[VN]], [[PTR]] scope:SCOPE_SYS
; ON-NEXT: global_atomic_add_u32 [[ZERO]], [[ONE]], [[PTR]] offset:4 scope:SCOPE_SYS
; ON-NEXT: global_atomic_add_u32 [[ZERO]], [[VN]], [[PTR]] offset:8 scope:SCOPE_SYS
; ON-NEXT: global_atomic_add_u32 [[ZERO]], [[VBASE]], [[PTR]] offset:12 scope:SCOPE_SYS
; ON-NEXT: s_wait_xcnt 0x0
; ON-NEXT: s_mov_b32 exec_lo, [[SAVE]]
; ON-NEXT: [[SKIP0]]:
; ON-NEXT: [[CLONE0:.Lpcsection[0-9]+]]:
; ON-NEXT: ds_store_b32 [[ADDR]], v{{[0-9]+}} offset:16
; The second site is inside divergent control flow.
; ON: s_cbranch_execz
; ON: v_readlane_b32
; ON: s_add_nc_u64 [[PTR2:s\[[0-9:]+\]]], [[PTR2]], k.prof.counters@rel64+24
; ON: [[CLONE1:.Lpcsection[0-9]+]]:
; ON-NEXT: ds_load_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:284
; ON: s_endpgm

; ON-O0-LABEL: {{^}}k.prof:
; ON-O0: v_readlane_b32
; ON-O0: s_add_nc_u64 {{.*}}, k.prof.counters@rel64+4
; ON-O0: global_atomic_add_u32
; ON-O0: ds_store_b32
; ON-O0: v_readlane_b32
; ON-O0: s_add_nc_u64 {{.*}}, k.prof.counters@rel64+24
; ON-O0: ds_load_b32
; ON-O0: k.prof.counters:
; ON-O0-NEXT: .zero 40

; The clone has the same LDS layout as the kernel.
; ON: .amdhsa_kernel k.prof
; ON: .amdhsa_group_segment_fixed_size 1024

; The clone numbered the same sequence of LDS instructions.
; ON: .type k.prof.ldsprof,@object
; ON-NEXT: k.prof.ldsprof:
; ON-NEXT: .long 2
; ON-NEXT: .long [[HASH]]
; ON-NEXT: .size k.prof.ldsprof, 8

; One record of 20 bytes per site.
; ON: .section .bss,"aw",@nobits
; ON: .globl k.prof.counters
; ON-NEXT: .protected k.prof.counters
; ON-NEXT: .type k.prof.counters,@object
; ON-NEXT: k.prof.counters:
; ON-NEXT: .zero 40
; ON-NEXT: .size k.prof.counters, 40

; ON: .section amdgpu_lds_prof_sites,"awo",@progbits,.text
; ON-NEXT: [[CBASE0:.Lpcsection_base[0-9]+]]:
; ON-NEXT: .long [[CLONE0]]-[[CBASE0]]
; ON-NEXT: .long 0
; ON-NEXT: .long 1
; ON-NEXT: [[CBASE1:.Lpcsection_base[0-9]+]]:
; ON-NEXT: .long [[CLONE1]]-[[CBASE1]]
; ON-NEXT: .long 1
; ON-NEXT: .long 1

; Both are kernels in the metadata.
; CHECK: .name: k
; CHECK: .symbol: k.kd
; ON: .name: k.prof
; ON: .symbol: k.prof.kd
define protected amdgpu_kernel void @k(ptr addrspace(1) %out, i32 %stride, i32 %cond) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %i = mul i32 %tid, %stride
  %im = and i32 %i, 127
  %p = getelementptr inbounds [256 x i32], ptr addrspace(3) @lds, i32 0, i32 %im
  %p4 = getelementptr inbounds i32, ptr addrspace(3) %p, i32 4
  store i32 %tid, ptr addrspace(3) %p4, align 4
  %c = icmp ugt i32 %tid, %cond
  br i1 %c, label %then, label %done

then:
  %j = add i32 %im, 71
  %jm = and i32 %j, 255
  %q = getelementptr inbounds [256 x i32], ptr addrspace(3) @lds, i32 0, i32 %jm
  %v = load i32, ptr addrspace(3) %q, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %tid
  store i32 %v, ptr addrspace(1) %o, align 4
  br label %done

done:
  ret void
}
