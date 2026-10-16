; RUN: llc -mtriple=hexagon-- -mattr=+hvxv75,+hvx-length128b,+hvx-qfloat,+hvx-ieee-fp \
; RUN:     -O2 < %s | FileCheck %s

; A multi-cycle scalar producer on SLOT2/SLOT3 (TC3x scalar multiply such as
; M2_mpysip, or TC4x scalar floating-point such as F2_sfmpy / F2_dfmpyhh) must
; not share a packet with a control-transfer instruction that implicitly reads
; the same (or an overlapping) register. The multi-cycle write commits after
; the transfer of control happens, so a co-packetized consumer would observe
; a stale ABI argument or return-value register. The def and the transfer
; must live in different packets. This applies uniformly to calls, tail calls
; (direct and indirect), and returns (PS_jmpret / L4_return / J2_jumpr r31).

; -----------------------------------------------------------------------------
; Scalar single-precision FP def feeding a call that reads the same GPR.

; CHECK-LABEL: mulredux_scalar_kernel:
; CHECK:      r0 = sfmpy(r0,r{{[0-9]+}})
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT: call __truncsfhf2
; CHECK-NEXT: }

define <64 x half> @mulredux_scalar_kernel(half %s, <64 x half> %v) {
._crit_edge:
  %m = fmul half %s, 0xH0000
  %b = bitcast half %m to <1 x half>
  %splat = shufflevector <1 x half> %b, <1 x half> zeroinitializer,
             <64 x i32> zeroinitializer
  %r = fmul <64 x half> %splat, %v
  ret <64 x half> %r
}

; -----------------------------------------------------------------------------
; Double-precision FP def (paired reg r1:r0) feeding a call that reads r0
; implicitly. The TC4x def of the pair overlaps the ABI argument register, so
; the two must not be co-packetized.

declare void @bar_i32(i32)

; CHECK-LABEL: dfmpy_paired_def_call:
; CHECK:      dfmpyhh(r{{[0-9]+}}:{{[0-9]+}},r{{[0-9]+}}:{{[0-9]+}})
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT: call bar_i32
; CHECK-NEXT: }

define void @dfmpy_paired_def_call(double %a, double %b) {
entry:
  %m = fmul double %a, %b
  %bits = bitcast double %m to i64
  %lo = trunc i64 %bits to i32
  call void @bar_i32(i32 %lo)
  ret void
}

; -----------------------------------------------------------------------------
; Scalar FP def followed by a tail call. The tail call is emitted as a
; jump-with-symbol and implicitly consumes the ABI argument register, so the
; same hazard applies.

declare void @bar_f32(float)

; CHECK-LABEL: sfmpy_tailcall:
; CHECK:      r0 = sfmpy(r0,r{{[0-9]+}})
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT: jump bar_f32
; CHECK-NEXT: }

define void @sfmpy_tailcall(float %a) {
  %m = fmul float %a, %a
  tail call void @bar_f32(float %m)
  ret void
}

; -----------------------------------------------------------------------------
; Scalar FP def of $r0 feeding a call whose implicit use is the wider $d0
; (r1:r0) i64 argument register. The def and the call overlap through $r0 and
; must not be co-packetized.

declare void @bar_i64(i64)

; CHECK-LABEL: sfmpy_narrow_def_wide_use:
; CHECK:      r0 = sfmpy(r0,r{{[0-9]+}})
; CHECK:      }
; CHECK:      {
; CHECK:      call bar_i64
; CHECK-NEXT: }

define void @sfmpy_narrow_def_wide_use(float %a, i32 %b) {
  %m = fmul float %a, %a
  %m_bits = bitcast float %m to i32
  %m64 = zext i32 %m_bits to i64
  %b64 = zext i32 %b to i64
  %hi = shl i64 %b64, 32
  %arg = or i64 %hi, %m64
  call void @bar_i64(i64 %arg)
  ret void
}

; -----------------------------------------------------------------------------
; Scalar FP def followed by an indirect tail call (J2_jumpr). The indirect
; branch consumes the ABI argument register implicitly and has the same
; stale-register hazard as a direct call.

; CHECK-LABEL: sfmpy_indirect_tailcall:
; CHECK:      r0 = sfmpy(r0,r{{[0-9]+}})
; CHECK:      }
; CHECK:      {
; CHECK-NEXT: {{call|jump}}r r{{[0-9]+}}
; CHECK-NEXT: }

define void @sfmpy_indirect_tailcall(float %a, ptr %fp) {
  %m = fmul float %a, %a
  tail call void %fp(float %m)
  ret void
}

; -----------------------------------------------------------------------------
; Scalar TC3x def (M2_mpysip / +mpyi) feeding a call that reads the same GPR.
; The TC3x write also completes late on real HW, so the multiply and the call
; must live in different packets.

declare void @bar_i32_2(i32)

; CHECK-LABEL: mpyi_scalar_call:
; CHECK:      r0 = {{[+]?}}mpyi(r0,#3)
; CHECK:      }
; CHECK:      {
; CHECK:      call bar_i32_2
; CHECK-NEXT: }

define void @mpyi_scalar_call(i32 %a) {
  %m = mul i32 %a, 3
  call void @bar_i32_2(i32 %m)
  ret void
}

; -----------------------------------------------------------------------------
; Scalar FP def producing the return value in the same packet as a PS_jmpret
; (which lowers to J2_jumpr r31 and implicitly reads $r0). Real HW would let
; the caller observe a stale return-value register when the def and the return
; are bundled, so they must live in different packets.

; CHECK-LABEL: sfmpy_return:
; CHECK:      r0 = sfmpy(r0,r{{[0-9]+}})
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT: jumpr r31
; CHECK-NEXT: }

define float @sfmpy_return(float %a, float %b) {
  %m = fmul float %a, %b
  ret float %m
}

; -----------------------------------------------------------------------------
; TC3x scalar multiply feeding the return-value register: same hazard as
; sfmpy_return, exercised for the TC3x path.

; CHECK-LABEL: mpyi_return:
; CHECK:      r0 = {{[+]?}}mpyi(r0,#3)
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT: jumpr r31
; CHECK-NEXT: }

define i32 @mpyi_return(i32 %a) {
  %m = mul i32 %a, 3
  ret i32 %m
}
