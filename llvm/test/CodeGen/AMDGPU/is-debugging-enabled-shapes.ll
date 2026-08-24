; RUN: split-file %s %t
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 %t/unused.ll -o - | FileCheck --check-prefixes=UNUSED %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 %t/store.ll -o - | FileCheck --check-prefixes=STORE %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 %t/return.ll -o - | FileCheck --check-prefixes=RETURN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 %t/select.ll -o - | FileCheck --check-prefixes=SELECT %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 %t/arithmetic.ll -o - | FileCheck --check-prefixes=ARITH %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 %t/multiple-branches.ll -o - | FileCheck --check-prefixes=MULTIBR %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 %t/branch-and-store.ll -o - | FileCheck --check-prefixes=BRSTORE %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 %t/ordered-placement.ll -o - | FileCheck --check-prefixes=ORDERED %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 %t/expect-ordered-placement.ll -o - | FileCheck --check-prefixes=EXPECTORDERED %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 %t/expect-multiple-uses.ll -o - | FileCheck --check-prefixes=EXPECTMULTI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 %t/store-between.ll -o - | FileCheck --check-prefixes=STOREBETWEEN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 %t/call-between.ll -o - | FileCheck --check-prefixes=CALLBETWEEN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 %t/two-observations.ll -o - | FileCheck --check-prefixes=TWOOBS %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 %t/negated-store-between.ll -o - | FileCheck --check-prefixes=NEGSTORE %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 %t/load-between.ll -o - | FileCheck --check-prefix=LOADBETWEEN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 %t/volatile-load-between.ll -o - | FileCheck --check-prefix=VOLLOADBETWEEN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 %t/barrier-between.ll -o - | FileCheck --check-prefix=BARRIERBETWEEN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 %t/fence-between.ll -o - | FileCheck --check-prefix=FENCEBETWEEN %s

; The same shapes through GlobalISel.
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 -global-isel -global-isel-abort=1 %t/unused.ll -o - | FileCheck --check-prefixes=UNUSED %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 -global-isel -global-isel-abort=1 %t/store.ll -o - | FileCheck --check-prefixes=STORE %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 -global-isel -global-isel-abort=1 %t/return.ll -o - | FileCheck --check-prefixes=RETURN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 -global-isel -global-isel-abort=1 %t/select.ll -o - | FileCheck --check-prefixes=SELECT %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 -global-isel -global-isel-abort=1 %t/arithmetic.ll -o - | FileCheck --check-prefixes=ARITH %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 -global-isel -global-isel-abort=1 %t/multiple-branches.ll -o - | FileCheck --check-prefixes=MULTIBR %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 -global-isel -global-isel-abort=1 %t/branch-and-store.ll -o - | FileCheck --check-prefixes=BRSTORE %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 -global-isel -global-isel-abort=1 %t/ordered-placement.ll -o - | FileCheck --check-prefixes=ORDERED %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 -global-isel -global-isel-abort=1 %t/expect-ordered-placement.ll -o - | FileCheck --check-prefixes=EXPECTORDERED %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 -global-isel -global-isel-abort=1 %t/expect-multiple-uses.ll -o - | FileCheck --check-prefixes=EXPECTMULTI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 -global-isel -global-isel-abort=1 %t/store-between.ll -o - | FileCheck --check-prefixes=STOREBETWEEN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 -global-isel -global-isel-abort=1 %t/call-between.ll -o - | FileCheck --check-prefixes=CALLBETWEEN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 -global-isel -global-isel-abort=1 %t/two-observations.ll -o - | FileCheck --check-prefixes=TWOOBS %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 -global-isel -global-isel-abort=1 %t/negated-store-between.ll -o - | FileCheck --check-prefixes=NEGSTORE %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 -global-isel -global-isel-abort=1 %t/load-between.ll -o - | FileCheck --check-prefix=LOADBETWEEN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 -global-isel -global-isel-abort=1 %t/volatile-load-between.ll -o - | FileCheck --check-prefix=VOLLOADBETWEEN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 -global-isel -global-isel-abort=1 %t/barrier-between.ll -o - | FileCheck --check-prefix=BARRIERBETWEEN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 -global-isel -global-isel-abort=1 %t/fence-between.ll -o - | FileCheck --check-prefix=FENCEBETWEEN %s

; On gfx11.5, where the register prints under its pre-GFX12 name.
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 %t/unused.ll -o - | FileCheck --check-prefixes=UNUSED %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 %t/store.ll -o - | FileCheck --check-prefixes=STORE %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 %t/return.ll -o - | FileCheck --check-prefixes=RETURN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 %t/select.ll -o - | FileCheck --check-prefixes=SELECT %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 %t/arithmetic.ll -o - | FileCheck --check-prefixes=ARITH %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 %t/multiple-branches.ll -o - | FileCheck --check-prefixes=MULTIBR %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 %t/branch-and-store.ll -o - | FileCheck --check-prefixes=BRSTORE %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 %t/ordered-placement.ll -o - | FileCheck --check-prefixes=ORDERED %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 %t/expect-ordered-placement.ll -o - | FileCheck --check-prefixes=EXPECTORDERED %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 %t/expect-multiple-uses.ll -o - | FileCheck --check-prefixes=EXPECTMULTI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 %t/store-between.ll -o - | FileCheck --check-prefixes=STOREBETWEEN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 %t/call-between.ll -o - | FileCheck --check-prefixes=CALLBETWEEN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 %t/two-observations.ll -o - | FileCheck --check-prefixes=TWOOBS %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 %t/negated-store-between.ll -o - | FileCheck --check-prefixes=NEGSTORE %s

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O0 %t/store.ll -o - | FileCheck --check-prefixes=STORE %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O0 -global-isel -global-isel-abort=1 %t/store.ll -o - | FileCheck --check-prefixes=STORE %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O0 %t/branch-and-store.ll -o - | FileCheck --check-prefixes=BRSTORE %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O0 -global-isel -global-isel-abort=1 %t/branch-and-store.ll -o - | FileCheck --check-prefixes=BRSTORE %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O0 %t/store-between.ll -o - | FileCheck --check-prefixes=STOREBETWEEN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O0 -global-isel -global-isel-abort=1 %t/store-between.ll -o - | FileCheck --check-prefixes=STOREBETWEEN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O0 %t/call-between.ll -o - | FileCheck --check-prefixes=CALLBETWEEN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O0 -global-isel -global-isel-abort=1 %t/call-between.ll -o - | FileCheck --check-prefixes=CALLBETWEEN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O0 %t/two-observations.ll -o - | FileCheck --check-prefixes=TWOOBS %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O0 -global-isel -global-isel-abort=1 %t/two-observations.ll -o - | FileCheck --check-prefixes=TWOOBS %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O0 %t/negated-store-between.ll -o - | FileCheck --check-prefixes=NEGSTORE %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O0 -global-isel -global-isel-abort=1 %t/negated-store-between.ll -o - | FileCheck --check-prefixes=NEGSTORE %s

; UNUSED-LABEL: unused:
; STORE-LABEL: store_result:
; RETURN-LABEL: return_result:
; SELECT-LABEL: select_result:
; MULTIBR-LABEL: multiple_branches:
; BRSTORE-LABEL: branch_and_store:
; EXPECTMULTI-LABEL: expect_multiple_uses:
; ORDERED-LABEL: ordered_placement:
; EXPECTORDERED-LABEL: expect_ordered_placement:
; STOREBETWEEN-LABEL: store_between:
; CALLBETWEEN-LABEL: call_between:
; TWOOBS-LABEL: two_observations:
; NEGSTORE-LABEL: negated_store_between:

; These either lack a sole branch use or cross an intervening side effect, so
; the query materializes instead of fusing. Note that llvm.sideeffect cannot
; serve as the intervening operation, because both instruction selectors drop
; it without leaving a node behind.
; UNUSED-NOT: s_cbranch_cdbgsys_or_user
; UNUSED: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_{{(WAVE_)?}}STATUS, 20, 2)
; UNUSED-NOT: s_cbranch_cdbgsys_or_user
; STORE-NOT: s_cbranch_cdbgsys_or_user
; STORE: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_{{(WAVE_)?}}STATUS, 20, 2)
; STORE-NOT: s_cbranch_cdbgsys_or_user
; RETURN-NOT: s_cbranch_cdbgsys_or_user
; RETURN: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_{{(WAVE_)?}}STATUS, 20, 2)
; RETURN-NOT: s_cbranch_cdbgsys_or_user
; SELECT-NOT: s_cbranch_cdbgsys_or_user
; SELECT: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_{{(WAVE_)?}}STATUS, 20, 2)
; SELECT-NOT: s_cbranch_cdbgsys_or_user
; MULTIBR-NOT: s_cbranch_cdbgsys_or_user
; MULTIBR: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_{{(WAVE_)?}}STATUS, 20, 2)
; MULTIBR-NOT: s_cbranch_cdbgsys_or_user
; BRSTORE-NOT: s_cbranch_cdbgsys_or_user
; BRSTORE: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_{{(WAVE_)?}}STATUS, 20, 2)
; BRSTORE-NOT: s_cbranch_cdbgsys_or_user
; EXPECTMULTI-NOT: s_cbranch_cdbgsys_or_user
; EXPECTMULTI: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_{{(WAVE_)?}}STATUS, 20, 2)
; EXPECTMULTI-NOT: s_cbranch_cdbgsys_or_user
; ORDERED-NOT: s_cbranch_cdbgsys_or_user
; ORDERED: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_{{(WAVE_)?}}STATUS, 20, 2)
; ORDERED-NOT: s_cbranch_cdbgsys_or_user
; EXPECTORDERED-NOT: s_cbranch_cdbgsys_or_user
; EXPECTORDERED: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_{{(WAVE_)?}}STATUS, 20, 2)
; EXPECTORDERED-NOT: s_cbranch_cdbgsys_or_user
; STOREBETWEEN-NOT: s_cbranch_cdbgsys_or_user
; STOREBETWEEN: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_{{(WAVE_)?}}STATUS, 20, 2)
; STOREBETWEEN-NOT: s_cbranch_cdbgsys_or_user
; CALLBETWEEN-NOT: s_cbranch_cdbgsys_or_user
; CALLBETWEEN: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_{{(WAVE_)?}}STATUS, 20, 2)
; CALLBETWEEN-NOT: s_cbranch_cdbgsys_or_user

; Fusing would swap two distinct observations, so both must
; materialize.
; TWOOBS-NOT: s_cbranch_cdbgsys_or_user
; TWOOBS-COUNT-2: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_{{(WAVE_)?}}STATUS, 20, 2)
; TWOOBS-NOT: s_cbranch_cdbgsys_or_user

; A negated query rejected by the scan must not lose its negation.
; NEGSTORE-NOT: s_cbranch_cdbgsys_or_user
; NEGSTORE: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_{{(WAVE_)?}}STATUS, 20, 2)
; NEGSTORE-NOT: s_cbranch_cdbgsys_or_user

; LOADBETWEEN-LABEL: load_between:
; LOADBETWEEN-NOT: s_cbranch_cdbgsys_or_user
; LOADBETWEEN: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_{{(WAVE_)?}}STATUS, 20, 2)
; LOADBETWEEN-NOT: s_cbranch_cdbgsys_or_user
; VOLLOADBETWEEN-LABEL: volatile_load_between:
; VOLLOADBETWEEN-NOT: s_cbranch_cdbgsys_or_user
; VOLLOADBETWEEN: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_{{(WAVE_)?}}STATUS, 20, 2)
; VOLLOADBETWEEN-NOT: s_cbranch_cdbgsys_or_user
; BARRIERBETWEEN-LABEL: barrier_between:
; BARRIERBETWEEN-NOT: s_cbranch_cdbgsys_or_user
; BARRIERBETWEEN: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_{{(WAVE_)?}}STATUS, 20, 2)
; BARRIERBETWEEN-NOT: s_cbranch_cdbgsys_or_user
; FENCEBETWEEN-LABEL: fence_between:
; FENCEBETWEEN-NOT: s_cbranch_cdbgsys_or_user
; FENCEBETWEEN: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_{{(WAVE_)?}}STATUS, 20, 2)
; FENCEBETWEEN-NOT: s_cbranch_cdbgsys_or_user

; ARITH-LABEL: arithmetic_use:

; A canonical negation feeding the sole conditional branch is fusable.
; ARITH-NOT: s_getreg_b32
; ARITH: s_cbranch_cdbgsys_or_user

;--- unused.ll
declare noundef i1 @llvm.is.debugging.enabled()

define amdgpu_kernel void @unused() {
  %enabled = call i1 @llvm.is.debugging.enabled()
  ret void
}

;--- store.ll
declare noundef i1 @llvm.is.debugging.enabled()

define amdgpu_kernel void @store_result(ptr addrspace(1) %out) {
  %enabled = call i1 @llvm.is.debugging.enabled()
  store i1 %enabled, ptr addrspace(1) %out
  ret void
}

;--- return.ll
declare noundef i1 @llvm.is.debugging.enabled()

define i1 @return_result() {
  %enabled = call i1 @llvm.is.debugging.enabled()
  ret i1 %enabled
}

;--- select.ll
declare noundef i1 @llvm.is.debugging.enabled()

define i32 @select_result() {
  %enabled = call i1 @llvm.is.debugging.enabled()
  %value = select i1 %enabled, i32 1, i32 0
  ret i32 %value
}

;--- arithmetic.ll
declare noundef i1 @llvm.is.debugging.enabled()
declare void @llvm.sideeffect()

define amdgpu_kernel void @arithmetic_use() {
entry:
  %enabled = call i1 @llvm.is.debugging.enabled()
  %disabled = xor i1 %enabled, true
  br i1 %disabled, label %normal, label %debug

debug:
  call void @llvm.sideeffect()
  br label %normal

normal:
  ret void
}

;--- load-between.ll
declare noundef i1 @llvm.is.debugging.enabled()

define amdgpu_kernel void @load_between(ptr addrspace(1) %in,
                                        ptr addrspace(1) %out) {
entry:
  %enabled = call i1 @llvm.is.debugging.enabled()
  %value = load i32, ptr addrspace(1) %in
  br i1 %enabled, label %debug, label %normal

debug:
  store volatile i32 %value, ptr addrspace(1) %out
  br label %exit

normal:
  %next = add i32 %value, 1
  store volatile i32 %next, ptr addrspace(1) %out
  br label %exit

exit:
  ret void
}

;--- volatile-load-between.ll
declare noundef i1 @llvm.is.debugging.enabled()

define amdgpu_kernel void @volatile_load_between(ptr addrspace(1) %in,
                                                 ptr addrspace(1) %out) {
entry:
  %enabled = call i1 @llvm.is.debugging.enabled()
  %value = load volatile i32, ptr addrspace(1) %in
  br i1 %enabled, label %debug, label %exit

debug:
  store volatile i32 %value, ptr addrspace(1) %out
  br label %exit

exit:
  ret void
}

;--- barrier-between.ll
declare noundef i1 @llvm.is.debugging.enabled()
declare void @llvm.amdgcn.s.barrier()

define amdgpu_kernel void @barrier_between(ptr addrspace(1) %out) {
entry:
  %enabled = call i1 @llvm.is.debugging.enabled()
  call void @llvm.amdgcn.s.barrier()
  br i1 %enabled, label %debug, label %exit

debug:
  store volatile i32 1, ptr addrspace(1) %out
  br label %exit

exit:
  ret void
}

;--- fence-between.ll
declare noundef i1 @llvm.is.debugging.enabled()

define amdgpu_kernel void @fence_between(ptr addrspace(1) %out) {
entry:
  %enabled = call i1 @llvm.is.debugging.enabled()
  fence syncscope("agent") seq_cst
  br i1 %enabled, label %debug, label %exit

debug:
  store volatile i32 1, ptr addrspace(1) %out
  br label %exit

exit:
  ret void
}

;--- multiple-branches.ll
declare noundef i1 @llvm.is.debugging.enabled()
declare void @llvm.sideeffect()

define amdgpu_kernel void @multiple_branches() {
entry:
  %enabled = call i1 @llvm.is.debugging.enabled()
  br i1 %enabled, label %again, label %normal

again:
  br i1 %enabled, label %debug, label %normal

debug:
  call void @llvm.sideeffect()
  br label %normal

normal:
  ret void
}

;--- branch-and-store.ll
declare noundef i1 @llvm.is.debugging.enabled()
declare void @llvm.sideeffect()

define amdgpu_kernel void @branch_and_store(ptr addrspace(1) %out) {
entry:
  %enabled = call i1 @llvm.is.debugging.enabled()
  store i1 %enabled, ptr addrspace(1) %out
  br i1 %enabled, label %debug, label %normal

debug:
  call void @llvm.sideeffect()
  br label %normal

normal:
  ret void
}

;--- ordered-placement.ll
declare noundef i1 @llvm.is.debugging.enabled()
declare void @llvm.sideeffect()

define amdgpu_kernel void @ordered_placement(ptr addrspace(1) %out) {
entry:
  %enabled = call i1 @llvm.is.debugging.enabled()
  store volatile i32 7, ptr addrspace(1) %out
  br i1 %enabled, label %debug, label %normal

debug:
  call void @llvm.sideeffect()
  br label %normal

normal:
  ret void
}

;--- expect-ordered-placement.ll
declare noundef i1 @llvm.is.debugging.enabled()
declare i1 @llvm.expect.i1(i1, i1 immarg)
declare void @llvm.sideeffect()

define amdgpu_kernel void @expect_ordered_placement(ptr addrspace(1) %out) {
entry:
  %raw = call i1 @llvm.is.debugging.enabled()
  %enabled = call i1 @llvm.expect.i1(i1 %raw, i1 false)
  store volatile i32 7, ptr addrspace(1) %out
  br i1 %enabled, label %debug, label %normal

debug:
  call void @llvm.sideeffect()
  br label %normal

normal:
  ret void
}

;--- expect-multiple-uses.ll
declare noundef i1 @llvm.is.debugging.enabled()
declare i1 @llvm.expect.i1(i1, i1 immarg)
declare void @llvm.sideeffect()

define amdgpu_kernel void @expect_multiple_uses(ptr addrspace(1) %out) {
entry:
  %raw = call i1 @llvm.is.debugging.enabled()
  %enabled = call i1 @llvm.expect.i1(i1 %raw, i1 false)
  store i1 %enabled, ptr addrspace(1) %out
  br i1 %enabled, label %debug, label %normal

debug:
  call void @llvm.sideeffect()
  br label %normal

normal:
  ret void
}

;--- store-between.ll
declare noundef i1 @llvm.is.debugging.enabled()

define amdgpu_kernel void @store_between(ptr addrspace(1) %out) {
entry:
  %enabled = call i1 @llvm.is.debugging.enabled()
  store volatile i32 7, ptr addrspace(1) %out
  br i1 %enabled, label %debug, label %normal

debug:
  store volatile i32 1, ptr addrspace(1) %out
  br label %normal

normal:
  ret void
}

;--- call-between.ll
declare noundef i1 @llvm.is.debugging.enabled()
declare void @ext()

define void @call_between(ptr addrspace(1) %out) {
entry:
  %enabled = call i1 @llvm.is.debugging.enabled()
  call void @ext()
  br i1 %enabled, label %debug, label %normal

debug:
  store volatile i32 1, ptr addrspace(1) %out
  br label %normal

normal:
  ret void
}

;--- two-observations.ll
declare noundef i1 @llvm.is.debugging.enabled()

define amdgpu_kernel void @two_observations(ptr addrspace(1) %out) {
entry:
  %first = call i1 @llvm.is.debugging.enabled()
  %second = call i1 @llvm.is.debugging.enabled()
  store volatile i1 %second, ptr addrspace(1) %out
  br i1 %first, label %debug, label %normal

debug:
  store volatile i32 1, ptr addrspace(1) %out
  br label %normal

normal:
  ret void
}

;--- negated-store-between.ll
declare noundef i1 @llvm.is.debugging.enabled()

define amdgpu_kernel void @negated_store_between(ptr addrspace(1) %out) {
entry:
  %enabled = call i1 @llvm.is.debugging.enabled()
  store volatile i32 7, ptr addrspace(1) %out
  %disabled = xor i1 %enabled, true
  br i1 %disabled, label %normal, label %debug

debug:
  store volatile i32 1, ptr addrspace(1) %out
  br label %normal

normal:
  ret void
}
