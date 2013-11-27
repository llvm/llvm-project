; RUN: llc < %s -march=sparcv9 -disable-sparc-delay-filler -disable-sparc-leaf-proc | FileCheck %s

; CHECK: intarg
; The save/restore frame is not strictly necessary here, but we would need to
; refer to %o registers instead.
; CHECK: save %sp, -128, %sp
; CHECK: stb %i0, [%i4]
; CHECK: stb %i1, [%i4]
; CHECK: sth %i2, [%i4]
; CHECK: st  %i3, [%i4]
; CHECK: stx %i4, [%i4]
; CHECK: st  %i5, [%i4]
; CHECK: ld [%fp+2227], [[R:%[gilo][0-7]]]
; CHECK: st  [[R]], [%i4]
; CHECK: ldx [%fp+2231], [[R:%[gilo][0-7]]]
; CHECK: stx [[R]], [%i4]
; CHECK: restore
define void @intarg(i8  %a0,   ; %i0
                    i8  %a1,   ; %i1
                    i16 %a2,   ; %i2
                    i32 %a3,   ; %i3
                    i8* %a4,   ; %i4
                    i32 %a5,   ; %i5
                    i32 signext %a6,   ; [%fp+BIAS+176]
                    i8* %a7) { ; [%fp+BIAS+184]
  store i8 %a0, i8* %a4
  store i8 %a1, i8* %a4
  %p16 = bitcast i8* %a4 to i16*
  store i16 %a2, i16* %p16
  %p32 = bitcast i8* %a4 to i32*
  store i32 %a3, i32* %p32
  %pp = bitcast i8* %a4 to i8**
  store i8* %a4, i8** %pp
  store i32 %a5, i32* %p32
  store i32 %a6, i32* %p32
  store i8* %a7, i8** %pp
  ret void
}

; CHECK: call_intarg
; 16 saved + 8 args.
; CHECK: save %sp, -192, %sp
; Sign-extend and store the full 64 bits.
; CHECK: sra %i0, 0, [[R:%[gilo][0-7]]]
; CHECK: stx [[R]], [%sp+2223]
; Use %o0-%o5 for outgoing arguments
; CHECK: or %g0, 5, %o5
; CHECK: call intarg
; CHECK-NOT: add %sp
; CHECK: restore
define void @call_intarg(i32 %i0, i8* %i1) {
  call void @intarg(i8 0, i8 1, i16 2, i32 3, i8* undef, i32 5, i32 %i0, i8* %i1)
  ret void
}

; CHECK: floatarg
; CHECK: save %sp, -128, %sp
; CHECK: fstod %f1,
; CHECK: faddd %f2,
; CHECK: faddd %f4,
; CHECK: faddd %f6,
; CHECK: ld [%fp+2307], [[F:%f[0-9]+]]
; CHECK: fadds %f31, [[F]]
define double @floatarg(float %a0,    ; %f1
                        double %a1,   ; %d2
                        double %a2,   ; %d4
                        double %a3,   ; %d6
                        float %a4,    ; %f9
                        float %a5,    ; %f11
                        float %a6,    ; %f13
                        float %a7,    ; %f15
                        float %a8,    ; %f17
                        float %a9,    ; %f19
                        float %a10,   ; %f21
                        float %a11,   ; %f23
                        float %a12,   ; %f25
                        float %a13,   ; %f27
                        float %a14,   ; %f29
                        float %a15,   ; %f31
                        float %a16,   ; [%fp+BIAS+256] (using 8 bytes)
                        double %a17) { ; [%fp+BIAS+264] (using 8 bytes)
  %d0 = fpext float %a0 to double
  %s1 = fadd double %a1, %d0
  %s2 = fadd double %a2, %s1
  %s3 = fadd double %a3, %s2
  %s16 = fadd float %a15, %a16
  %d16 = fpext float %s16 to double
  %s17 = fadd double %d16, %s3
  ret double %s17
}

; CHECK: call_floatarg
; CHECK: save %sp, -272, %sp
; Store 4 bytes, right-aligned in slot.
; CHECK: st %f1, [%sp+2307]
; Store 8 bytes in full slot.
; CHECK: std %f2, [%sp+2311]
; CHECK: fmovd %f2, %f4
; CHECK: call floatarg
; CHECK-NOT: add %sp
; CHECK: restore
define void @call_floatarg(float %f1, double %d2, float %f5, double *%p) {
  %r = call double @floatarg(float %f5, double %d2, double %d2, double %d2,
                             float %f5, float %f5,  float %f5,  float %f5,
                             float %f5, float %f5,  float %f5,  float %f5,
                             float %f5, float %f5,  float %f5,  float %f5,
                             float %f1, double %d2)
  store double %r, double* %p
  ret void
}

; CHECK: mixedarg
; CHECK: fstod %f3
; CHECK: faddd %f6
; CHECK: faddd %f16
; CHECK: ldx [%fp+2231]
; CHECK: ldx [%fp+2247]
define void @mixedarg(i8 %a0,      ; %i0
                      float %a1,   ; %f3
                      i16 %a2,     ; %i2
                      double %a3,  ; %d6
                      i13 %a4,     ; %i4
                      float %a5,   ; %f11
                      i64 %a6,     ; [%fp+BIAS+176]
                      double *%a7, ; [%fp+BIAS+184]
                      double %a8,  ; %d16
                      i16* %a9) {  ; [%fp+BIAS+200]
  %d1 = fpext float %a1 to double
  %s3 = fadd double %a3, %d1
  %s8 = fadd double %a8, %s3
  store double %s8, double* %a7
  store i16 %a2, i16* %a9
  ret void
}

; CHECK: call_mixedarg
; CHECK: stx %i2, [%sp+2247]
; CHECK: stx %i0, [%sp+2223]
; CHECK: fmovd %f2, %f6
; CHECK: fmovd %f2, %f16
; CHECK: call mixedarg
; CHECK-NOT: add %sp
; CHECK: restore
define void @call_mixedarg(i64 %i0, double %f2, i16* %i2) {
  call void @mixedarg(i8 undef,
                      float undef,
                      i16 undef,
                      double %f2,
                      i13 undef,
                      float undef,
                      i64 %i0,
                      double* undef,
                      double %f2,
                      i16* %i2)
  ret void
}

; The inreg attribute is used to indicate 32-bit sized struct elements that
; share an 8-byte slot.
; CHECK: inreg_fi
; CHECK: fstoi %f1
; CHECK: srlx %i0, 32, [[R:%[gilo][0-7]]]
; CHECK: sub [[R]],
define i32 @inreg_fi(i32 inreg %a0,     ; high bits of %i0
                     float inreg %a1) { ; %f1
  %b1 = fptosi float %a1 to i32
  %rv = sub i32 %a0, %b1
  ret i32 %rv
}

; CHECK: call_inreg_fi
; Allocate space for 6 arguments, even when only 2 are used.
; CHECK: save %sp, -176, %sp
; CHECK: sllx %i1, 32, %o0
; CHECK: fmovs %f5, %f1
; CHECK: call inreg_fi
define void @call_inreg_fi(i32* %p, i32 %i1, float %f5) {
  %x = call i32 @inreg_fi(i32 %i1, float %f5)
  ret void
}

; CHECK: inreg_ff
; CHECK: fsubs %f0, %f1, %f1
define float @inreg_ff(float inreg %a0,   ; %f0
                       float inreg %a1) { ; %f1
  %rv = fsub float %a0, %a1
  ret float %rv
}

; CHECK: call_inreg_ff
; CHECK: fmovs %f3, %f0
; CHECK: fmovs %f5, %f1
; CHECK: call inreg_ff
define void @call_inreg_ff(i32* %p, float %f3, float %f5) {
  %x = call float @inreg_ff(float %f3, float %f5)
  ret void
}

; CHECK: inreg_if
; CHECK: fstoi %f0
; CHECK: sub %i0
define i32 @inreg_if(float inreg %a0, ; %f0
                     i32 inreg %a1) { ; low bits of %i0
  %b0 = fptosi float %a0 to i32
  %rv = sub i32 %a1, %b0
  ret i32 %rv
}

; CHECK: call_inreg_if
; CHECK: fmovs %f3, %f0
; CHECK: or %g0, %i2, %o0
; CHECK: call inreg_if
define void @call_inreg_if(i32* %p, float %f3, i32 %i2) {
  %x = call i32 @inreg_if(float %f3, i32 %i2)
  ret void
}

; The frontend shouldn't do this. Just pass i64 instead.
; CHECK: inreg_ii
; CHECK: srlx %i0, 32, [[R:%[gilo][0-7]]]
; CHECK: sub %i0, [[R]], %i0
define i32 @inreg_ii(i32 inreg %a0,   ; high bits of %i0
                     i32 inreg %a1) { ; low bits of %i0
  %rv = sub i32 %a1, %a0
  ret i32 %rv
}

; CHECK: call_inreg_ii
; CHECK: srl %i2, 0, [[R2:%[gilo][0-7]]]
; CHECK: sllx %i1, 32, [[R1:%[gilo][0-7]]]
; CHECK: or [[R1]], [[R2]], %o0
; CHECK: call inreg_ii
define void @call_inreg_ii(i32* %p, i32 %i1, i32 %i2) {
  %x = call i32 @inreg_ii(i32 %i1, i32 %i2)
  ret void
}

; Structs up to 32 bytes in size can be returned in registers.
; CHECK: ret_i64_pair
; CHECK: ldx [%i2], %i0
; CHECK: ldx [%i3], %i1
define { i64, i64 } @ret_i64_pair(i32 %a0, i32 %a1, i64* %p, i64* %q) {
  %r1 = load i64* %p
  %rv1 = insertvalue { i64, i64 } undef, i64 %r1, 0
  store i64 0, i64* %p
  %r2 = load i64* %q
  %rv2 = insertvalue { i64, i64 } %rv1, i64 %r2, 1
  ret { i64, i64 } %rv2
}

; CHECK: call_ret_i64_pair
; CHECK: call ret_i64_pair
; CHECK: stx %o0, [%i0]
; CHECK: stx %o1, [%i0]
define void @call_ret_i64_pair(i64* %i0) {
  %rv = call { i64, i64 } @ret_i64_pair(i32 undef, i32 undef,
                                        i64* undef, i64* undef)
  %e0 = extractvalue { i64, i64 } %rv, 0
  store i64 %e0, i64* %i0
  %e1 = extractvalue { i64, i64 } %rv, 1
  store i64 %e1, i64* %i0
  ret void
}

; This is not a C struct, each member uses 8 bytes.
; CHECK: ret_i32_float_pair
; CHECK: ld [%i2], %i0
; CHECK: ld [%i3], %f3
define { i32, float } @ret_i32_float_pair(i32 %a0, i32 %a1,
                                          i32* %p, float* %q) {
  %r1 = load i32* %p
  %rv1 = insertvalue { i32, float } undef, i32 %r1, 0
  store i32 0, i32* %p
  %r2 = load float* %q
  %rv2 = insertvalue { i32, float } %rv1, float %r2, 1
  ret { i32, float } %rv2
}

; CHECK: call_ret_i32_float_pair
; CHECK: call ret_i32_float_pair
; CHECK: st %o0, [%i0]
; CHECK: st %f3, [%i1]
define void @call_ret_i32_float_pair(i32* %i0, float* %i1) {
  %rv = call { i32, float } @ret_i32_float_pair(i32 undef, i32 undef,
                                                i32* undef, float* undef)
  %e0 = extractvalue { i32, float } %rv, 0
  store i32 %e0, i32* %i0
  %e1 = extractvalue { i32, float } %rv, 1
  store float %e1, float* %i1
  ret void
}

; This is a C struct, each member uses 4 bytes.
; CHECK: ret_i32_float_packed
; CHECK: ld [%i2], [[R:%[gilo][0-7]]]
; CHECK: sllx [[R]], 32, %i0
; CHECK: ld [%i3], %f1
define inreg { i32, float } @ret_i32_float_packed(i32 %a0, i32 %a1,
                                                  i32* %p, float* %q) {
  %r1 = load i32* %p
  %rv1 = insertvalue { i32, float } undef, i32 %r1, 0
  store i32 0, i32* %p
  %r2 = load float* %q
  %rv2 = insertvalue { i32, float } %rv1, float %r2, 1
  ret { i32, float } %rv2
}

; CHECK: call_ret_i32_float_packed
; CHECK: call ret_i32_float_packed
; CHECK: srlx %o0, 32, [[R:%[gilo][0-7]]]
; CHECK: st [[R]], [%i0]
; CHECK: st %f1, [%i1]
define void @call_ret_i32_float_packed(i32* %i0, float* %i1) {
  %rv = call { i32, float } @ret_i32_float_packed(i32 undef, i32 undef,
                                                  i32* undef, float* undef)
  %e0 = extractvalue { i32, float } %rv, 0
  store i32 %e0, i32* %i0
  %e1 = extractvalue { i32, float } %rv, 1
  store float %e1, float* %i1
  ret void
}

; The C frontend should use i64 to return { i32, i32 } structs, but verify that
; we don't miscompile thi case where both struct elements are placed in %i0.
; CHECK: ret_i32_packed
; CHECK: ld [%i2], [[R1:%[gilo][0-7]]]
; CHECK: ld [%i3], [[R2:%[gilo][0-7]]]
; CHECK: sllx [[R2]], 32, [[R3:%[gilo][0-7]]]
; CHECK: or [[R3]], [[R1]], %i0
define inreg { i32, i32 } @ret_i32_packed(i32 %a0, i32 %a1,
                                          i32* %p, i32* %q) {
  %r1 = load i32* %p
  %rv1 = insertvalue { i32, i32 } undef, i32 %r1, 1
  store i32 0, i32* %p
  %r2 = load i32* %q
  %rv2 = insertvalue { i32, i32 } %rv1, i32 %r2, 0
  ret { i32, i32 } %rv2
}

; CHECK: call_ret_i32_packed
; CHECK: call ret_i32_packed
; CHECK: srlx %o0, 32, [[R:%[gilo][0-7]]]
; CHECK: st [[R]], [%i0]
; CHECK: st %o0, [%i1]
define void @call_ret_i32_packed(i32* %i0, i32* %i1) {
  %rv = call { i32, i32 } @ret_i32_packed(i32 undef, i32 undef,
                                          i32* undef, i32* undef)
  %e0 = extractvalue { i32, i32 } %rv, 0
  store i32 %e0, i32* %i0
  %e1 = extractvalue { i32, i32 } %rv, 1
  store i32 %e1, i32* %i1
  ret void
}

; The return value must be sign-extended to 64 bits.
; CHECK: ret_sext
; CHECK: sra %i0, 0, %i0
define signext i32 @ret_sext(i32 %a0) {
  ret i32 %a0
}

; CHECK: ret_zext
; CHECK: srl %i0, 0, %i0
define zeroext i32 @ret_zext(i32 %a0) {
  ret i32 %a0
}

; CHECK: ret_nosext
; CHECK-NOT: sra
define signext i32 @ret_nosext(i32 signext %a0) {
  ret i32 %a0
}

; CHECK: ret_nozext
; CHECK-NOT: srl
define signext i32 @ret_nozext(i32 signext %a0) {
  ret i32 %a0
}

; CHECK-LABEL: test_register_directive
; CHECK:       .register %g2, #scratch
; CHECK:       .register %g3, #scratch
; CHECK:       add %i0, 2, %g2
; CHECK:       add %i0, 3, %g3
define i32 @test_register_directive(i32 %i0) {
entry:
  %0 = add nsw i32 %i0, 2
  %1 = add nsw i32 %i0, 3
  tail call void asm sideeffect "", "r,r,~{l0},~{l1},~{l2},~{l3},~{l4},~{l5},~{l6},~{l7},~{i0},~{i1},~{i2},~{i3},~{i4},~{i5},~{i6},~{i7},~{o0},~{o1},~{o2},~{o3},~{o4},~{o5},~{o6},~{o7},~{g1},~{g4},~{g5},~{g6},~{g7}"(i32 %0, i32 %1)
  %2 = add nsw i32 %0, %1
  ret i32 %2
}

; CHECK-LABEL: test_large_stack

; CHECK:       sethi 16, %g1
; CHECK:       xor %g1, -176, %g1
; CHECK:       save %sp, %g1, %sp

; CHECK:       sethi 14, %g1
; CHECK:       xor %g1, -1, %g1
; CHECK:       add %g1, %fp, %g1
; CHECK:       call use_buf

define i32 @test_large_stack() {
entry:
  %buffer1 = alloca [16384 x i8], align 8
  %buffer1.sub = getelementptr inbounds [16384 x i8]* %buffer1, i32 0, i32 0
  %0 = call i32 @use_buf(i32 16384, i8* %buffer1.sub)
  ret i32 %0
}

declare i32 @use_buf(i32, i8*)
