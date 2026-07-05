; RUN: llc < %s -mtriple=armv7-apple-ios -verify-machineinstrs | FileCheck %s --check-prefix=CHECK --check-prefix CHECK-ARMV7
; RUN: llc < %s -mtriple=thumbv7-apple-ios -verify-machineinstrs | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-T2
; RUN: llc < %s -mtriple=thumbv6-apple-ios -verify-machineinstrs | FileCheck %s --check-prefix=CHECK-T1
; RUN: llc < %s -mtriple=thumbv6-apple-ios -verify-machineinstrs -mcpu=cortex-m0 | FileCheck %s --check-prefix=CHECK-T1-M0
; RUN: llc < %s -mtriple=thumbv7--none-eabi -thread-model single -verify-machineinstrs | FileCheck %s --check-prefix=CHECK-BAREMETAL

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

; CHECK-LABEL: _func:
define void @func(i32 %argc, ptr %argv) nounwind {
entry:
	%argc.addr = alloca i32		; <ptr> [#uses=1]
	%argv.addr = alloca ptr		; <ptr> [#uses=1]
	%val1 = alloca i32		; <ptr> [#uses=2]
	%val2 = alloca i32		; <ptr> [#uses=15]
	%andt = alloca i32		; <ptr> [#uses=2]
	%ort = alloca i32		; <ptr> [#uses=2]
	%xort = alloca i32		; <ptr> [#uses=2]
	%old = alloca i32		; <ptr> [#uses=18]
	%temp = alloca i32		; <ptr> [#uses=2]
	store i32 %argc, ptr %argc.addr
	store ptr %argv, ptr %argv.addr
	store i32 0, ptr %val1
	store i32 31, ptr %val2
	store i32 3855, ptr %andt
	store i32 3855, ptr %ort
	store i32 3855, ptr %xort
	store i32 4, ptr %temp
	%tmp = load i32, ptr %temp
	call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"()
  ; CHECK: ldrex
  ; CHECK: add
  ; CHECK: strex
  ; CHECK-T1: bl ___sync_fetch_and_add_4
  ; CHECK-T1-M0: bl ___atomic_fetch_add_4
  ; CHECK-BAREMETAL: add
  ; CHECK-BAREMETAL-NOT: __sync
  %0 = atomicrmw add ptr %val1, i32 %tmp monotonic
	store i32 %0, ptr %old
	call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"()
  ; CHECK: ldrex
  ; CHECK: sub
  ; CHECK: strex
  ; CHECK-T1: bl ___sync_fetch_and_sub_4
  ; CHECK-T1-M0: bl ___atomic_fetch_sub_4
  ; CHECK-BAREMETAL: sub
  ; CHECK-BAREMETAL-NOT: __sync
  %1 = atomicrmw sub ptr %val2, i32 30 monotonic
	store i32 %1, ptr %old
	call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"()
  ; CHECK: ldrex
  ; CHECK: add
  ; CHECK: strex
  ; CHECK-T1: bl ___sync_fetch_and_add_4
  ; CHECK-T1-M0: bl ___atomic_fetch_add_4
  ; CHECK-BAREMETAL: add
  ; CHECK-BAREMETAL-NOT: __sync
  %2 = atomicrmw add ptr %val2, i32 1 monotonic
	store i32 %2, ptr %old
	call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"()
  ; CHECK: ldrex
  ; CHECK: sub
  ; CHECK: strex
  ; CHECK-T1: bl ___sync_fetch_and_sub_4
  ; CHECK-T1-M0: bl ___atomic_fetch_sub_4
  ; CHECK-BAREMETAL: sub
  ; CHECK-BAREMETAL-NOT: __sync
  %3 = atomicrmw sub ptr %val2, i32 1 monotonic
	store i32 %3, ptr %old
	call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"()
  ; CHECK: ldrex
  ; CHECK: and
  ; CHECK: strex
  ; CHECK-T1: bl ___sync_fetch_and_and_4
  ; CHECK-T1-M0: bl ___atomic_fetch_and_4
  ; CHECK-BAREMETAL: and
  ; CHECK-BAREMETAL-NOT: __sync
  %4 = atomicrmw and ptr %andt, i32 4080 monotonic
	store i32 %4, ptr %old
	call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"()
  ; CHECK: ldrex
  ; CHECK: or
  ; CHECK: strex
  ; CHECK-T1: bl ___sync_fetch_and_or_4
  ; CHECK-T1-M0: bl ___atomic_fetch_or_4
  ; CHECK-BAREMETAL: or
  ; CHECK-BAREMETAL-NOT: __sync
  %5 = atomicrmw or ptr %ort, i32 4080 monotonic
	store i32 %5, ptr %old
	call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"()
  ; CHECK: ldrex
  ; CHECK: eor
  ; CHECK: strex
  ; CHECK-T1: bl ___sync_fetch_and_xor_4
  ; CHECK-T1-M0: bl ___atomic_fetch_xor_4
  ; CHECK-BAREMETAL: eor
  ; CHECK-BAREMETAL-NOT: __sync
  %6 = atomicrmw xor ptr %xort, i32 4080 monotonic
	store i32 %6, ptr %old
	call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"()
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  ; CHECK-T1: bl ___sync_fetch_and_min_4
  ; CHECK-T1-M0: bl ___atomic_compare_exchange_4
  ; CHECK-BAREMETAL: cmp
  ; CHECK-BAREMETAL-NOT: __sync
  %7 = atomicrmw min ptr %val2, i32 16 monotonic
	store i32 %7, ptr %old
	call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"()
	%neg = sub i32 0, 1
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  ; CHECK-T1: bl ___sync_fetch_and_min_4
  ; CHECK-T1-M0: bl ___atomic_compare_exchange_4
  ; CHECK-BAREMETAL: cmp
  ; CHECK-BAREMETAL-NOT: __sync
  %8 = atomicrmw min ptr %val2, i32 %neg monotonic
	store i32 %8, ptr %old
	call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"()
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  ; CHECK-T1: bl ___sync_fetch_and_max_4
  ; CHECK-T1-M0: bl ___atomic_compare_exchange_4
  ; CHECK-BAREMETAL: cmp
  ; CHECK-BAREMETAL-NOT: __sync
  %9 = atomicrmw max ptr %val2, i32 1 monotonic
	store i32 %9, ptr %old
	call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"()
  ; CHECK: ldrex
  ; CHECK: bic
  ; CHECK-NOT: cmp
  ; CHECK: strex
  ; CHECK-T1: bl ___sync_fetch_and_max_4
  ; CHECK-T1-M0: bl ___atomic_compare_exchange_4
  ; CHECK-BAREMETAL: bic
  ; CHECK-BAREMETAL-NOT: __sync
  %10 = atomicrmw max ptr %val2, i32 0 monotonic
	store i32 %10, ptr %old
	call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"()
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  ; CHECK-T1: bl ___sync_fetch_and_umin_4
  ; CHECK-T1-M0: bl ___atomic_compare_exchange_4
  ; CHECK-BAREMETAL: cmp
  ; CHECK-BAREMETAL-NOT: __sync
  %11 = atomicrmw umin ptr %val2, i32 16 monotonic
	store i32 %11, ptr %old
	call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"()
	%uneg = sub i32 0, 1
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  ; CHECK-T1: bl ___sync_fetch_and_umin_4
  ; CHECK-T1-M0: bl ___atomic_compare_exchange_4
  ; CHECK-BAREMETAL: cmp
  ; CHECK-BAREMETAL-NOT: __sync
  %12 = atomicrmw umin ptr %val2, i32 %uneg monotonic
	store i32 %12, ptr %old
	call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"()
  ; CHECK: ldrex
  ; CHECK: strex
  ; CHECK: cmp
  ; CHECK-T1: bl ___sync_fetch_and_umax_4
  ; CHECK-T1-M0: bl ___atomic_compare_exchange_4
  ; CHECK-BAREMETAL: cmp
  ; CHECK-BAREMETAL-NOT: __sync
  %13 = atomicrmw umax ptr %val2, i32 1 monotonic
	store i32 %13, ptr %old
	call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"()
  ; CHECK: ldrex
  ; CHECK: strex
  ; CHECK: cmp
  ; CHECK-T1: bl ___sync_fetch_and_umax_4
  ; CHECK-T1-M0: bl ___atomic_compare_exchange_4
  ; CHECK-BAREMETAL: cmp
  ; CHECK-BAREMETAL-NOT: __sync
  %14 = atomicrmw umax ptr %val2, i32 0 monotonic
	store i32 %14, ptr %old

  ret void
}

; CHECK-LABEL: _func2:
define void @func2() nounwind {
entry:
  %val = alloca i16
  %old = alloca i16
  store i16 31, ptr %val
  ; CHECK: ldrex
  ; CHECK: strex
  ; CHECK: cmp
  ; CHECK-T1: bl ___sync_fetch_and_umin_2
  ; CHECK-T1-M0: bl ___atomic_compare_exchange_2
  ; CHECK-BAREMETAL: cmp
  ; CHECK-BAREMETAL-NOT: __sync
  %0 = atomicrmw umin ptr %val, i16 16 monotonic
  store i16 %0, ptr %old
  %uneg = sub i16 0, 1
  ; CHECK: ldrex
  ; CHECK: strex
  ; CHECK: cmp
  ; CHECK-T1: bl ___sync_fetch_and_umin_2
  ; CHECK-T1-M0: bl ___atomic_compare_exchange_2
  ; CHECK-BAREMETAL: cmp
  ; CHECK-BAREMETAL-NOT: __sync
  %1 = atomicrmw umin ptr %val, i16 %uneg monotonic
  store i16 %1, ptr %old
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  ; CHECK-T1: bl ___sync_fetch_and_umax_2
  ; CHECK-T1-M0: bl ___atomic_compare_exchange_2
  ; CHECK-BAREMETAL: cmp
  ; CHECK-BAREMETAL-NOT: __sync
  %2 = atomicrmw umax ptr %val, i16 1 monotonic
  store i16 %2, ptr %old
  ; CHECK: ldrex
  ; CHECK: strex
  ; CHECK: cmp
  ; CHECK-T1: bl ___sync_fetch_and_umax_2
  ; CHECK-T1-M0: bl ___atomic_compare_exchange_2
  ; CHECK-BAREMETAL: cmp
  ; CHECK-BAREMETAL-NOT: __sync
  %3 = atomicrmw umax ptr %val, i16 0 monotonic
  store i16 %3, ptr %old
  ret void
}

; CHECK-LABEL: _func3:
define void @func3() nounwind {
entry:
  %val = alloca i8
  %old = alloca i8
  store i8 31, ptr %val
  ; CHECK: ldrex
  ; CHECK: strex
  ; CHECK: cmp
  ; CHECK-T1: bl ___sync_fetch_and_umin_1
  ; CHECK-T1-M0: bl ___atomic_compare_exchange_1
  ; CHECK-BAREMETAL: cmp
  ; CHECK-BAREMETAL-NOT: __sync
  %0 = atomicrmw umin ptr %val, i8 16 monotonic
  store i8 %0, ptr %old
  ; CHECK: ldrex
  ; CHECK: strex
  ; CHECK: cmp
  ; CHECK-T1: bl ___sync_fetch_and_umin_1
  ; CHECK-T1-M0: bl ___atomic_compare_exchange_1
  ; CHECK-BAREMETAL: cmp
  ; CHECK-BAREMETAL-NOT: __sync
  %uneg = sub i8 0, 1
  %1 = atomicrmw umin ptr %val, i8 %uneg monotonic
  store i8 %1, ptr %old
  ; CHECK: ldrex
  ; CHECK: strex
  ; CHECK: cmp
  ; CHECK-T1: bl ___sync_fetch_and_umax_1
  ; CHECK-T1-M0: bl ___atomic_compare_exchange_1
  ; CHECK-BAREMETAL: cmp
  ; CHECK-BAREMETAL-NOT: __sync
  %2 = atomicrmw umax ptr %val, i8 1 monotonic
  store i8 %2, ptr %old
  ; CHECK: ldrex
  ; CHECK: strex
  ; CHECK: cmp
  ; CHECK-T1: bl ___sync_fetch_and_umax_1
  ; CHECK-T1-M0: bl ___atomic_compare_exchange_1
  ; CHECK-BAREMETAL: cmp
  ; CHECK-BAREMETAL-NOT: __sync
  %3 = atomicrmw umax ptr %val, i8 0 monotonic
  store i8 %3, ptr %old
  ret void
}

; CHECK-LABEL: _func4:
; This function should not need to use callee-saved registers.
; rdar://problem/12203728
; CHECK-NOT: r4
define i32 @func4(ptr %p) nounwind optsize ssp {
entry:
  %0 = atomicrmw add ptr %p, i32 1 monotonic
  ret i32 %0
}

define i32 @test_cmpxchg_fail_order(ptr %addr, i32 %desired, i32 %new) {
; CHECK-LABEL: test_cmpxchg_fail_order:

  %pair = cmpxchg ptr %addr, i32 %desired, i32 %new seq_cst monotonic
  %oldval = extractvalue { i32, i1 } %pair, 0
; CHECK-ARMV7:     mov     r[[ADDR:[0-9]+]], r0
; CHECK-ARMV7:     ldrex   [[OLDVAL:r[0-9]+]], [r0]
; CHECK-ARMV7:     cmp     [[OLDVAL]], r1
; CHECK-ARMV7:     bne     [[FAIL_BB:\.?LBB[0-9]+_[0-9]+]]
; CHECK-ARMV7:     dmb ish
; CHECK-ARMV7: [[LOOP_BB:\.?LBB.*]]:
; CHECK-ARMV7:     strex   [[SUCCESS:r[0-9]+]], r2, [r[[ADDR]]]
; CHECK-ARMV7:     cmp     [[SUCCESS]], #0
; CHECK-ARMV7:     beq     [[SUCCESS_BB:\.?LBB.*]]
; CHECK-ARMV7:     ldrex   [[OLDVAL]], [r[[ADDR]]]
; CHECK-ARMV7:     cmp     [[OLDVAL]], r1
; CHECK-ARMV7:     beq     [[LOOP_BB]]
; CHECK-ARMV7: [[FAIL_BB]]:
; CHECK-ARMV7:     clrex
; CHECK-ARMV7:     bx      lr
; CHECK-ARMV7: [[SUCCESS_BB]]:
; CHECK-ARMV7:     dmb     ish
; CHECK-ARMV7:     bx      lr

; CHECK-T2:     mov     r[[ADDR:[0-9]+]], r0
; CHECK-T2:     ldrex   [[OLDVAL:r[0-9]+]], [r0]
; CHECK-T2:     cmp     [[OLDVAL]], r1
; CHECK-T2:     bne     [[FAIL_BB:\.?LBB.*]]
; CHECK-T2:     dmb ish
; CHECK-T2: [[LOOP_BB:\.?LBB.*]]:
; CHECK-T2:     strex   [[SUCCESS:r[0-9]+]], r2, [r[[ADDR]]]
; CHECK-T2:     cmp     [[SUCCESS]], #0
; CHECK-T2:     dmbeq   ish
; CHECK-T2:     bxeq    lr
; CHECK-T2:     ldrex   [[OLDVAL]], [r[[ADDR]]]
; CHECK-T2:     cmp     [[OLDVAL]], r1
; CHECK-T2:     beq     [[LOOP_BB]]
; CHECK-T2:     clrex

  ret i32 %oldval
}

define i32 @test_cmpxchg_fail_order1(ptr %addr, i32 %desired, i32 %new) {
; CHECK-LABEL: test_cmpxchg_fail_order1:

  %pair = cmpxchg ptr %addr, i32 %desired, i32 %new acquire acquire
  %oldval = extractvalue { i32, i1 } %pair, 0
; CHECK-NOT:     dmb ish
; CHECK: [[LOOP_BB:\.?LBB[0-9]+_1]]:
; CHECK:     ldrex   [[OLDVAL:r[0-9]+]], [r[[ADDR:[0-9]+]]]
; CHECK:     cmp     [[OLDVAL]], r1
; CHECK:     bne     [[FAIL_BB:\.?LBB[0-9]+_[0-9]+]]
; CHECK:     strex   [[SUCCESS:r[0-9]+]], r2, [r[[ADDR]]]
; CHECK:     cmp     [[SUCCESS]], #0
; CHECK:     bne     [[LOOP_BB]]
; CHECK:     dmb     ish
; CHECK:     bx      lr
; CHECK: [[FAIL_BB]]:
; CHECK-NEXT: clrex
; CHECK:     dmb     ish
; CHECK:     bx      lr

  ret i32 %oldval
}

define i32 @load_load_add_acquire(ptr %mem1, ptr %mem2) nounwind {
; CHECK-LABEL: load_load_add_acquire
  %val1 = load atomic i32, ptr %mem1 acquire, align 4
  %val2 = load atomic i32, ptr %mem2 acquire, align 4
  %tmp = add i32 %val1, %val2

; CHECK: ldr {{r[0-9]}}, [r0]
; CHECK: dmb
; CHECK: ldr {{r[0-9]}}, [r1]
; CHECK: dmb
; CHECK: add r0,

; CHECK-T1-M0: __atomic_load_4
; CHECK-T1-M0: __atomic_load_4

; CHECK-T1: ___sync_val_compare_and_swap_4
; CHECK-T1: ___sync_val_compare_and_swap_4

; CHECK-BAREMETAL: ldr {{r[0-9]}}, [r0]
; CHECK-BAREMETAL-NOT: dmb
; CHECK-BAREMETAL: ldr {{r[0-9]}}, [r1]
; CHECK-BAREMETAL-NOT: dmb
; CHECK-BAREMETAL: add r0,

  ret i32 %tmp
}

define void @store_store_release(ptr %mem1, i32 %val1, ptr %mem2, i32 %val2) {
; CHECK-LABEL: store_store_release
  store atomic i32 %val1, ptr %mem1 release, align 4
  store atomic i32 %val2, ptr %mem2 release, align 4

; CHECK: dmb
; CHECK: str r1, [r0]
; CHECK: dmb
; CHECK: str r3, [r2]

; CHECK-T1: ___sync_lock_test_and_set
; CHECK-T1: ___sync_lock_test_and_set

; CHECK-T1-M0: __atomic_store_4
; CHECK-T1-M0: __atomic_store_4

; CHECK-BAREMETAL-NOT: dmb
; CHECK-BAREMETAL: str r1, [r0]
; CHECK-BAREMETAL-NOT: dmb
; CHECK-BAREMETAL: str r3, [r2]

  ret void
}

define void @load_fence_store_monotonic(ptr %mem1, ptr %mem2) {
; CHECK-LABEL: load_fence_store_monotonic
  %val = load atomic i32, ptr %mem1 monotonic, align 4
  fence seq_cst
  store atomic i32 %val, ptr %mem2 monotonic, align 4

; CHECK: ldr [[R0:r[0-9]]], [r0]
; CHECK: dmb
; CHECK: str [[R0]], [r1]

; CHECK-T1-M0: __atomic_load_4
; CHECK-T1-M0: dmb
; CHECK-T1-M0: __atomic_store_4

; CHECK-T1: ldr [[R0:r[0-9]]], [{{r[0-9]+}}]
; CHECK-T1: {{dmb|bl ___sync_synchronize}}
; CHECK-T1: str [[R0]], [{{r[0-9]+}}]

; CHECK-BAREMETAL: ldr [[R0:r[0-9]]], [r0]
; CHECK-BAREMETAL-NOT: dmb
; CHECK-BAREMETAL: str [[R0]], [r1]

  ret void
}
