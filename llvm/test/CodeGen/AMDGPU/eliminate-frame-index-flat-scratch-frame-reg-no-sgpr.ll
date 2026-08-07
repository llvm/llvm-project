; RUN: not --crash llc -O3 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; NOTE: SIRegisterInfo::eliminateFrameIndex currently asserts on a
; flat-scratch target when a frame index used by a VALU instruction reaches the
; generic SGPR scavenging path, the function has a frame register, and no SGPR
; is free for the scavenger. Control then enters the branch that assumes no
; frame register exists: it asserts !FrameReg and its SVS lowering materializes
; the address from the offset alone, without a frame-register term.
;
; The reproducer needs all of following conditions:
;   1. A VALU frame-index user (V_SUB_CO_U32_e32 from the addrspacecast).
;   2. A frame register (non-entry function with a stack object).
;   3. A non-zero frame offset (large stack arg area from the many arguments).
;   4. No free SGPR for the scavenger (all pinned by the inline asm).
;   5. A flat-scratch target (gfx942/gfx950).
; The data dependency through %diff keeps every SGPR live across the subtract so
; the scheduler cannot free one up.

; CHECK: there is a frame register!

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define fastcc i64 @no_scavengeable_sgpr_with_frame_register(
    ptr %p0, i32 %i0, i64 %l0, i64 %l1, i64 %l2,
    i64 %l3, i64 %l4, i16 %s0, i32 %i1, ptr %p1,
    i64 %l5, i32 %i2, i32 %i3, i32 %i4,
    double %d0, double %d1, double %d2,
    i64 %l6, i64 %l7, i64 %l8, i64 %l9
) #0 {
entry:
  %local = alloca i64, align 8, addrspace(5)

  ; Fill all 16 allocatable SGPRs (8 pairs).
  %asm = call {i64, i64, i64, i64, i64, i64, i64, i64}
    asm sideeffect "; fill sgprs",
    "={s[0:1]},={s[2:3]},={s[4:5]},={s[6:7]},={s[8:9]},={s[10:11]},={s[12:13]},={s[14:15]}"()

  %s01 = extractvalue {i64, i64, i64, i64, i64, i64, i64, i64} %asm, 0
  %s23 = extractvalue {i64, i64, i64, i64, i64, i64, i64, i64} %asm, 1
  %s45 = extractvalue {i64, i64, i64, i64, i64, i64, i64, i64} %asm, 2
  %s67 = extractvalue {i64, i64, i64, i64, i64, i64, i64, i64} %asm, 3
  %s89 = extractvalue {i64, i64, i64, i64, i64, i64, i64, i64} %asm, 4
  %s1011 = extractvalue {i64, i64, i64, i64, i64, i64, i64, i64} %asm, 5
  %s1213 = extractvalue {i64, i64, i64, i64, i64, i64, i64, i64} %asm, 6
  %s1415 = extractvalue {i64, i64, i64, i64, i64, i64, i64, i64} %asm, 7

  ; The addrspacecast + subtract produces V_SUB_CO_U32_e32 with %stack.0.
  %flat_ptr = addrspacecast ptr addrspace(5) %local to ptr
  %ptrint = ptrtoint ptr %flat_ptr to i64
  %loaded = load i64, ptr null, align 8
  %diff = sub i64 %ptrint, %loaded

  ; Use every SGPR value together with the subtract result so the "use all" asm
  ; stays after the subtract, keeping all SGPRs live across it.
  call void asm sideeffect "; use all",
    "{s[0:1]},{s[2:3]},{s[4:5]},{s[6:7]},{s[8:9]},{s[10:11]},{s[12:13]},{s[14:15]},{v[0:1]}"(
    i64 %s01, i64 %s23, i64 %s45, i64 %s67,
    i64 %s89, i64 %s1011, i64 %s1213, i64 %s1415,
    i64 %diff)

  ret i64 0
}

attributes #0 = { "amdgpu-num-sgpr"="16" }