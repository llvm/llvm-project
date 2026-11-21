; RUN: llc -mtriple=hexagon -mcpu=hexagonv85 -mattr=+hlx -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=hexagon -mcpu=hexagonv85 -mattr=+hlx -verify-machineinstrs -stop-after=prolog-epilog < %s | FileCheck %s --check-prefix=PEI

; preserve_hlx functions may have HLX types in their signature. HLX arguments
; stay callee-saved; the HLX return registers are not.

;; Caller side: call-preserved masks.

; HLX arguments without an HLX return value keep the full preserve_hlx mask.
; ISEL-LABEL: name: call_args_void
; ISEL: J2_call @callee_args_void, hexagoncsr_preservehlx,
define void @call_args_void(<1024 x i16> %a, <1024 x i16> %b) {
  call hexagon_preserve_hlx void @callee_args_void(<1024 x i16> %a, <1024 x i16> %b)
  ret void
}

; ISEL-LABEL: name: call_args_scalar_ret
; ISEL: J2_call @callee_args_scalar_ret, hexagoncsr_preservehlx,
define i32 @call_args_scalar_ret(<1024 x i16> %a) {
  %r = call hexagon_preserve_hlx i32 @callee_args_scalar_ret(<1024 x i16> %a)
  ret i32 %r
}

; An HLX return value in L0 removes L0 and its super-register LL0.
; ISEL-LABEL: name: call_ret
; ISEL: J2_call @callee_ret, CustomRegMask($d8,$d9,$d10,$d11,$d12,$d13,$l1,{{.*}},$l15,$ll1,
define <1024 x i16> @call_ret(<1024 x i16> %a, <1024 x i16> %b) {
  %r = call hexagon_preserve_hlx <1024 x i16> @callee_ret(<1024 x i16> %a)
  %s = add <1024 x i16> %r, %b
  ret <1024 x i16> %s
}

; An HLX pair return value in LL0 removes LL0 and its sub-registers L0 and L1.
; ISEL-LABEL: name: call_ret_pair
; ISEL: J2_call @callee_ret_pair, CustomRegMask($d8,$d9,$d10,$d11,$d12,$d13,$l2,{{.*}},$l15,$ll1,
define <2048 x i16> @call_ret_pair(<2048 x i16> %a) {
  %r = call hexagon_preserve_hlx <2048 x i16> @callee_ret_pair(<2048 x i16> %a)
  ret <2048 x i16> %r
}

; ISEL-LABEL: name: call_ret_indirect
; ISEL: J2_callr {{.*}}, CustomRegMask($d8,$d9,$d10,$d11,$d12,$d13,$l1,{{.*}},$l15,$ll1,
define <1024 x i16> @call_ret_indirect(ptr %fp, <1024 x i16> %a) {
  %r = call hexagon_preserve_hlx <1024 x i16> %fp(<1024 x i16> %a)
  ret <1024 x i16> %r
}

; A discarded HLX return value is still clobbered.
; ISEL-LABEL: name: call_discard_ret
; ISEL: J2_call @callee_ret, CustomRegMask($d8,$d9,$d10,$d11,$d12,$d13,$l1,{{.*}},$l15,$ll1,
define void @call_discard_ret(<1024 x i16> %a) {
  call hexagon_preserve_hlx <1024 x i16> @callee_ret(<1024 x i16> %a)
  ret void
}

;; Tail calls. Only a preserve_hlx caller can tail call a preserve_hlx callee.

; The caller returns the same type, so it clobbers the same register. Nothing
; is live across the tail call, so it keeps the static mask (as on X86).
; ISEL-LABEL: name: tail_ret
; ISEL: PS_tailcall_i @callee_ret, hexagoncsr_preservehlx,
define hexagon_preserve_hlx <1024 x i16> @tail_ret(<1024 x i16> %a) {
  %r = tail call hexagon_preserve_hlx <1024 x i16> @callee_ret(<1024 x i16> %a)
  ret <1024 x i16> %r
}

; ISEL-LABEL: name: tail_args_void
; ISEL: PS_tailcall_i @callee_args_void, hexagoncsr_preservehlx,
define hexagon_preserve_hlx void @tail_args_void(<1024 x i16> %a) {
  tail call hexagon_preserve_hlx void @callee_args_void(<1024 x i16> %a)
  ret void
}

; The caller returns void, so it preserves L0, but the callee clobbers L0 as
; its return register. This must not be a tail call.
; ISEL-LABEL: name: tail_discard_ret
; ISEL-NOT: PS_tailcall_i
; ISEL: J2_call @callee_ret, CustomRegMask($d8,$d9,$d10,$d11,$d12,$d13,$l1,{{.*}},$l15,$ll1,
define hexagon_preserve_hlx void @tail_discard_ret(<1024 x i16> %a) {
  %r = tail call hexagon_preserve_hlx <1024 x i16> @callee_ret(<1024 x i16> %a)
  ret void
}

; The incoming argument L0 is saved but still read by the call, so the spill
; must not kill it.
; PEI-LABEL: name: tail_discard_ret
; PEI: PS_storerl {{.*}}$l0 ::
; PEI: J2_call @callee_ret
; PEI: $l0 = PS_loadrl

; The incoming pair LL0 is saved as L0 and L1 but still read by the call.
; PEI-LABEL: name: live_arg_pair
; PEI: PS_storerl {{.*}}$l0 ::
; PEI: PS_storerl {{.*}}$l1 ::
; PEI: J2_call @callee_ret_pair
; PEI: $l0 = PS_loadrl
; PEI: $l1 = PS_loadrl
define hexagon_preserve_hlx void @live_arg_pair(<2048 x i16> %a) {
  call hexagon_preserve_hlx <2048 x i16> @callee_ret_pair(<2048 x i16> %a)
  ret void
}

; The calling conventions differ, so this is not a tail call.
; ISEL-LABEL: name: tail_from_c
; ISEL-NOT: PS_tailcall_i
; ISEL: J2_call @callee_args_void, hexagoncsr_preservehlx,
define void @tail_from_c(<1024 x i16> %a) {
  tail call hexagon_preserve_hlx void @callee_args_void(<1024 x i16> %a)
  ret void
}

; HLX argument registers are preserved, so the caller keeps %a and %b in
; registers across the call.
; PEI-LABEL: name: call_args_void_live
; PEI-NOT: PS_storer
; PEI: J2_call @callee_args_void
; PEI-NOT: PS_loadr
; PEI: LX_add_h
define <1024 x i16> @call_args_void_live(<1024 x i16> %a, <1024 x i16> %b) {
  call hexagon_preserve_hlx void @callee_args_void(<1024 x i16> %a, <1024 x i16> %b)
  %s = add <1024 x i16> %a, %b
  ret <1024 x i16> %s
}

;; Callee side: saved registers.

; Clobbered HLX argument registers are saved and restored.
; PEI-LABEL: name: callee_args_void
; PEI: PS_storerl {{.*}}$l0 ::
; PEI: PS_storerl {{.*}}$l1 ::
; PEI: $l0 = PS_loadrl
; PEI: $l1 = PS_loadrl
define hexagon_preserve_hlx void @callee_args_void(<1024 x i16> %a, <1024 x i16> %b) {
  call void asm sideeffect "", "~{l0},~{l1}"()
  ret void
}

; The return register L0 is not saved; the clobbered L1 is.
; PEI-LABEL: name: callee_ret
; PEI-NOT: {{\$(l0|ll0) ::|\$(l0|ll0) = PS_loadr}}
; PEI: PS_storerl {{.*}}$l1 ::
; PEI-NOT: {{\$(l0|ll0) ::|\$(l0|ll0) = PS_loadr}}
; PEI: $l1 = PS_loadrl
; PEI-NOT: {{\$(l0|ll0) ::|\$(l0|ll0) = PS_loadr}}
; PEI: L4_return
define hexagon_preserve_hlx <1024 x i16> @callee_ret(<1024 x i16> %a) {
  call void asm sideeffect "", "~{l0},~{l1}"()
  %r = add <1024 x i16> %a, %a
  ret <1024 x i16> %r
}

; The return register LL0 (L0, L1) is not saved; the clobbered L2 is.
; PEI-LABEL: name: callee_ret_pair
; PEI-NOT: {{\$(l0|l1|ll0) ::|\$(l0|l1|ll0) = PS_loadr}}
; PEI: PS_storerl {{.*}}$l2 ::
; PEI-NOT: {{\$(l0|l1|ll0) ::|\$(l0|l1|ll0) = PS_loadr}}
; PEI: $l2 = PS_loadrl
; PEI-NOT: {{\$(l0|l1|ll0) ::|\$(l0|l1|ll0) = PS_loadr}}
; PEI: L4_return
define hexagon_preserve_hlx <2048 x i16> @callee_ret_pair(<2048 x i16> %a) {
  call void asm sideeffect "", "~{l0},~{l1},~{l2}"()
  ret <2048 x i16> %a
}

; __builtin_eh_return saves all callee-saved registers, but still not the
; return register L0.
; PEI-LABEL: name: callee_ret_eh
; PEI-NOT: {{\$(l0|ll0) ::|\$(l0|ll0) = PS_loadr}}
; PEI: EH_RETURN_JMPR
; PEI-NOT: {{\$(l0|ll0) ::|\$(l0|ll0) = PS_loadr}}
; PEI: L4_return
define hexagon_preserve_hlx <1024 x i16> @callee_ret_eh(<1024 x i16> %a, i1 %c, i32 %off, ptr %h) {
  call void asm sideeffect "", "~{l0},~{l1}"()
  br i1 %c, label %eh, label %ret
eh:
  call void @llvm.eh.return.i32(i32 %off, ptr %h)
  unreachable
ret:
  %r = add <1024 x i16> %a, %a
  ret <1024 x i16> %r
}

declare hexagon_preserve_hlx i32 @callee_args_scalar_ret(<1024 x i16>)
declare void @llvm.eh.return.i32(i32, ptr)
