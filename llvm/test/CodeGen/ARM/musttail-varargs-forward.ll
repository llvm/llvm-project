; RUN: llc -verify-machineinstrs -mtriple=thumbv7-unknown-windows-msvc %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=armv7-unknown-linux-gnueabi %s -o - | FileCheck %s

; A variadic function forwarding its arguments with a musttail call must keep the
; unnamed argument registers live, or the scratch register for the callee address
; is taken from r1-r3 and clobbers an argument the tail call still needs.

; Unnamed arguments occupy r1-r3, so the scratch must come from outside them.
define arm_aapcs_vfpcc void @fwd_variadic(ptr %this, ...) nounwind {
; CHECK-LABEL: fwd_variadic:
; CHECK: ldr{{(\.w)?}} r12, [r0]
; CHECK: ldr{{(\.w)?}} r12, [r12, #4]
; CHECK: bx r12
  %vt = load ptr, ptr %this, align 4
  %slot = getelementptr inbounds ptr, ptr %vt, i32 1
  %fn = load ptr, ptr %slot, align 4
  musttail call arm_aapcs_vfpcc void (ptr, ...) %fn(ptr %this, ...)
  ret void
}

; One named argument in r1; the unnamed window is r2-r3.
define arm_aapcs_vfpcc void @fwd_variadic_named1(ptr %this, i32 %n, ...) nounwind {
; CHECK-LABEL: fwd_variadic_named1:
; CHECK: ldr{{(\.w)?}} r12, [r0]
; CHECK: ldr{{(\.w)?}} r12, [r12, #4]
; CHECK: bx r12
  %vt = load ptr, ptr %this, align 4
  %slot = getelementptr inbounds ptr, ptr %vt, i32 1
  %fn = load ptr, ptr %slot, align 4
  musttail call arm_aapcs_vfpcc void (ptr, i32, ...) %fn(ptr %this, i32 %n, ...)
  ret void
}

; Two values live at once; the ABI does not fix which registers replace them.
define arm_aapcs_vfpcc void @fwd_variadic_two_scratch(ptr %this, ...) nounwind {
; CHECK-LABEL: fwd_variadic_two_scratch:
; CHECK-NOT: ldr r1,
; CHECK-NOT: ldr r2,
; CHECK-NOT: ldr r3,
; Anchored to end-of-line: a bare "bx r1" would also match "bx r12".
; CHECK-NOT: bx r1{{$}}
; CHECK-NOT: bx r2{{$}}
; CHECK-NOT: bx r3{{$}}
; CHECK: bx {{r([4-9]|1[012])}}
  %vt = load ptr, ptr %this, align 4
  %s1 = getelementptr inbounds ptr, ptr %vt, i32 1
  %f1 = load ptr, ptr %s1, align 4
  %s2 = getelementptr inbounds ptr, ptr %vt, i32 5
  %f2 = load ptr, ptr %s2, align 4
  %c = icmp eq ptr %f1, null
  %fn = select i1 %c, ptr %f2, ptr %f1
  musttail call arm_aapcs_vfpcc void (ptr, ...) %fn(ptr %this, ...)
  ret void
}

; Control: non-variadic, always compiled correctly; output must not change.
define arm_aapcs_vfpcc void @fwd_nonvariadic_all4(ptr %this, i32 %a, i32 %b, i32 %c) nounwind {
; CHECK-LABEL: fwd_nonvariadic_all4:
; CHECK: ldr{{(\.w)?}} r12, [r0]
; CHECK: ldr{{(\.w)?}} r12, [r12, #4]
; CHECK: bx r12
  %vt = load ptr, ptr %this, align 4
  %slot = getelementptr inbounds ptr, ptr %vt, i32 1
  %fn = load ptr, ptr %slot, align 4
  musttail call arm_aapcs_vfpcc void %fn(ptr %this, i32 %a, i32 %b, i32 %c)
  ret void
}

; The MSVC virtual-inheritance thunks are variadic musttail forwarders too; see
; clang/test/CodeGenCXX/ms-thunks-unprototyped.cpp.

declare arm_aapcs_vfpcc void @adjusted_target(ptr, ...)
declare arm_aapcs_vfpcc void @adjusted_target_ex(ptr, ...)
declare arm_aapcs_vfpcc void @adjusted_target_nv(ptr, i32, i32, i32)

; vtordisp: one load, one subtract, one direct tail branch.
define arm_aapcs_vfpcc void @vtordisp_thunk(ptr %this, ...) nounwind {
; CHECK-LABEL: vtordisp_thunk:
; CHECK:       ldr{{(\.w)?}} r12, [r0, #-4]
; CHECK-NOT:   ldr{{(\.w)?}} r1,
; CHECK-NOT:   ldr{{(\.w)?}} r2,
; CHECK-NOT:   ldr{{(\.w)?}} r3,
; CHECK:       sub{{(\.w)?}} r0, r0, r12
; CHECK:       b{{(\.w)?}} adjusted_target
  %offp = getelementptr inbounds i8, ptr %this, i32 -4
  %off = load i32, ptr %offp, align 4
  %negoff = sub i32 0, %off
  %adj = getelementptr i8, ptr %this, i32 %negoff
  musttail call arm_aapcs_vfpcc void (ptr, ...) @adjusted_target(ptr %adj, ...)
  ret void
}

; vtordispex: two values live at once, so pin the first load and forbid the
; argument registers for the rest.
define arm_aapcs_vfpcc void @vtordispex_thunk(ptr %this, ...) nounwind {
; CHECK-LABEL: vtordispex_thunk:
; CHECK:       ldr{{(\.w)?}} r12, [r0, #-4]
; CHECK-NOT:   ldr{{(\.w)?}} r1,
; CHECK-NOT:   ldr{{(\.w)?}} r2,
; CHECK-NOT:   ldr{{(\.w)?}} r3,
; CHECK:       b{{(\.w)?}} adjusted_target_ex
  %offp = getelementptr inbounds i8, ptr %this, i32 -4
  %off = load i32, ptr %offp, align 4
  %negoff = sub i32 0, %off
  %adj = getelementptr i8, ptr %this, i32 %negoff
  %vbptrp = getelementptr inbounds i8, ptr %adj, i32 -8
  %vbtable = load ptr, ptr %vbptrp, align 4
  %vbslot = getelementptr inbounds i32, ptr %vbtable, i32 2
  %vbase = load i32, ptr %vbslot, align 4
  %adj2 = getelementptr inbounds i8, ptr %vbptrp, i32 %vbase
  %adj3 = getelementptr i8, ptr %adj2, i32 8
  musttail call arm_aapcs_vfpcc void (ptr, ...) @adjusted_target_ex(ptr %adj3, ...)
  ret void
}

; Control: the same adjustment, non-variadic.  Output must not change.
define arm_aapcs_vfpcc void @vtordisp_nonvariadic_control(ptr %this, i32 %a, i32 %b, i32 %c) nounwind {
; CHECK-LABEL: vtordisp_nonvariadic_control:
; CHECK:       ldr{{(\.w)?}} r12, [r0, #-4]
; CHECK-NOT:   ldr{{(\.w)?}} r1,
; CHECK-NOT:   ldr{{(\.w)?}} r2,
; CHECK-NOT:   ldr{{(\.w)?}} r3,
; CHECK:       sub{{(\.w)?}} r0, r0, r12
; CHECK:       b{{(\.w)?}} adjusted_target_nv
  %offp = getelementptr inbounds i8, ptr %this, i32 -4
  %off = load i32, ptr %offp, align 4
  %negoff = sub i32 0, %off
  %adj = getelementptr i8, ptr %this, i32 %negoff
  musttail call arm_aapcs_vfpcc void @adjusted_target_nv(ptr %adj, i32 %a, i32 %b, i32 %c)
  ret void
}
