; RUN: llc -march=mipsel -mcpu=mips32 -mattr=+fp64,+msa,-nooddspreg \
; RUN:     -no-integrated-as < %s | FileCheck %s -check-prefix=ALL \
; RUN:     -check-prefix=ODDSPREG
; RUN: llc -march=mipsel -mcpu=mips32 -mattr=+fp64,+msa,+nooddspreg \
; RUN:     -no-integrated-as < %s | FileCheck %s -check-prefix=ALL \
; RUN:     -check-prefix=NOODDSPREG

@v4f32 = global <4 x float> zeroinitializer

define void @msa_insert_0(float %a) {
entry:
  ; Force the float into an odd-numbered register using named registers and
  ; load the vector.
  %b = call float asm sideeffect "mov.s $0, $1", "={$f13},{$f12}" (float %a)
  %0 = load volatile <4 x float>, <4 x float>* @v4f32

  ; Clobber all except $f12/$w12 and $f13
  ;
  ; The intention is that if odd single precision registers are permitted, the
  ; allocator will choose $f12/$w12 for the vector and $f13 for the float to
  ; avoid the spill/reload.
  ;
  ; On the other hand, if odd single precision registers are not permitted, it
  ; must copy $f13 to an even-numbered register before inserting into the
  ; vector.
  call void asm sideeffect "teqi $$zero, 1", "~{$f0},~{$f1},~{$f2},~{$f3},~{$f4},~{$f5},~{$f6},~{$f7},~{$f8},~{$f9},~{$f10},~{$f11},~{$f14},~{$f15},~{$f16},~{$f17},~{$f18},~{$f19},~{$f20},~{$f21},~{$f22},~{$f23},~{$f24},~{$f25},~{$f26},~{$f27},~{$f28},~{$f29},~{$f30},~{$f31}"()
  %1 = insertelement <4 x float> %0, float %b, i32 0
  store <4 x float> %1, <4 x float>* @v4f32
  ret void
}

; ALL-LABEL:  msa_insert_0:
; ALL:            mov.s $f13, $f12
; ALL:            lw $[[R0:[0-9]+]], %got(v4f32)(
; ALL:            ld.w $w[[W0:[0-9]+]], 0($[[R0]])
; NOODDSPREG:     mov.s $f[[F0:[0-9]+]], $f13
; NOODDSPREG:     insve.w $w[[W0]][0], $w[[F0]][0]
; ODDSPREG:       insve.w $w[[W0]][0], $w13[0]
; ALL:            teqi $zero, 1
; ALL-NOT: sdc1
; ALL-NOT: ldc1
; ALL:            st.w $w[[W0]], 0($[[R0]])

define void @msa_insert_1(float %a) {
entry:
  ; Force the float into an odd-numbered register using named registers and
  ; load the vector.
  %b = call float asm sideeffect "mov.s $0, $1", "={$f13},{$f12}" (float %a)
  %0 = load volatile <4 x float>, <4 x float>* @v4f32

  ; Clobber all except $f12/$w12 and $f13
  ;
  ; The intention is that if odd single precision registers are permitted, the
  ; allocator will choose $f12/$w12 for the vector and $f13 for the float to
  ; avoid the spill/reload.
  ;
  ; On the other hand, if odd single precision registers are not permitted, it
  ; must copy $f13 to an even-numbered register before inserting into the
  ; vector.
  call void asm sideeffect "teqi $$zero, 1", "~{$f0},~{$f1},~{$f2},~{$f3},~{$f4},~{$f5},~{$f6},~{$f7},~{$f8},~{$f9},~{$f10},~{$f11},~{$f14},~{$f15},~{$f16},~{$f17},~{$f18},~{$f19},~{$f20},~{$f21},~{$f22},~{$f23},~{$f24},~{$f25},~{$f26},~{$f27},~{$f28},~{$f29},~{$f30},~{$f31}"()
  %1 = insertelement <4 x float> %0, float %b, i32 1
  store <4 x float> %1, <4 x float>* @v4f32
  ret void
}

; ALL-LABEL:  msa_insert_1:
; ALL:            mov.s $f13, $f12
; ALL:            lw $[[R0:[0-9]+]], %got(v4f32)(
; ALL:            ld.w $w[[W0:[0-9]+]], 0($[[R0]])
; NOODDSPREG:     mov.s $f[[F0:[0-9]+]], $f13
; NOODDSPREG:     insve.w $w[[W0]][1], $w[[F0]][0]
; ODDSPREG:       insve.w $w[[W0]][1], $w13[0]
; ALL:            teqi $zero, 1
; ALL-NOT: sdc1
; ALL-NOT: ldc1
; ALL:            st.w $w[[W0]], 0($[[R0]])

define float @msa_extract_0() {
entry:
  %0 = load volatile <4 x float>, <4 x float>* @v4f32
  %1 = call <4 x float> asm sideeffect "move.v $0, $1", "={$w13},{$w12}" (<4 x float> %0)

  ; Clobber all except $f12, and $f13
  ;
  ; The intention is that if odd single precision registers are permitted, the
  ; allocator will choose $f13/$w13 for the vector since that saves on moves.
  ;
  ; On the other hand, if odd single precision registers are not permitted, it
  ; must move it to $f12/$w12.
  call void asm sideeffect "teqi $$zero, 1", "~{$f0},~{$f1},~{$f2},~{$f3},~{$f4},~{$f5},~{$f6},~{$f7},~{$f8},~{$f9},~{$f10},~{$f11},~{$f14},~{$f15},~{$f16},~{$f17},~{$f18},~{$f19},~{$f20},~{$f21},~{$f22},~{$f23},~{$f24},~{$f25},~{$f26},~{$f27},~{$f28},~{$f29},~{$f30},~{$f31}"()

  %2 = extractelement <4 x float> %1, i32 0
  ret float %2
}

; ALL-LABEL:  msa_extract_0:
; ALL:            lw $[[R0:[0-9]+]], %got(v4f32)(
; ALL:            ld.w $w12, 0($[[R0]])
; ALL:            move.v $w[[W0:13]], $w12
; NOODDSPREG:     move.v $w[[W0:12]], $w13
; ALL:            teqi $zero, 1
; ALL-NOT: st.w
; ALL-NOT: ld.w
; ALL:            mov.s $f0, $f[[W0]]

define float @msa_extract_1() {
entry:
  %0 = load volatile <4 x float>, <4 x float>* @v4f32
  %1 = call <4 x float> asm sideeffect "move.v $0, $1", "={$w13},{$w12}" (<4 x float> %0)

  ; Clobber all except $f13
  ;
  ; The intention is that if odd single precision registers are permitted, the
  ; allocator will choose $f13/$w13 for the vector since that saves on moves.
  ;
  ; On the other hand, if odd single precision registers are not permitted, it
  ; must be spilled.
  call void asm sideeffect "teqi $$zero, 1", "~{$f0},~{$f1},~{$f2},~{$f3},~{$f4},~{$f5},~{$f6},~{$f7},~{$f8},~{$f9},~{$f10},~{$f11},~{$f12},~{$f14},~{$f15},~{$f16},~{$f17},~{$f18},~{$f19},~{$f20},~{$f21},~{$f22},~{$f23},~{$f24},~{$f25},~{$f26},~{$f27},~{$f28},~{$f29},~{$f30},~{$f31}"()

  %2 = extractelement <4 x float> %1, i32 1
  ret float %2
}

; ALL-LABEL:  msa_extract_1:
; ALL:            lw $[[R0:[0-9]+]], %got(v4f32)(
; ALL:            ld.w $w12, 0($[[R0]])
; ALL:            splati.w $w[[W0:[0-9]+]], $w13[1]
; NOODDSPREG:     st.w $w[[W0]], 0($sp)
; ODDSPREG-NOT: st.w
; ODDSPREG-NOT: ld.w
; ALL:            teqi $zero, 1
; ODDSPREG-NOT: st.w
; ODDSPREG-NOT: ld.w
; NOODDSPREG:     ld.w $w0, 0($sp)
; ODDSPREG:       mov.s $f0, $f[[W0]]
