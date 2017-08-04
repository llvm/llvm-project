; RUN: llc < %s -march=sparcv9 -disable-sparc-delay-filler -disable-sparc-leaf-proc | FileCheck %s --check-prefix=CHECK --check-prefix=HARD
; RUN: llc < %s -march=sparcv9 -disable-sparc-delay-filler -disable-sparc-leaf-proc -mattr=soft-float | FileCheck %s --check-prefix=CHECK --check-prefix=SOFT

; CHECK-LABEL: intarg:
; The save/restore frame is not strictly necessary here, but we would need to
; refer to %o registers instead.
; CHECK: save %sp, -128, %sp
; CHECK: ldx [%fp+2231], [[R2:%[gilo][0-7]]]
; CHECK: ld [%fp+2227], [[R1:%[gilo][0-7]]]
; CHECK: stb %i0, [%i4]
; CHECK: stb %i1, [%i4]
; CHECK: sth %i2, [%i4]
; CHECK: st  %i3, [%i4]
; CHECK: stx %i4, [%i4]
; CHECK: st  %i5, [%i4]
; CHECK: st  [[R1]], [%i4]
; CHECK: stx [[R2]], [%i4]
; CHECK: restore
define void @intarg(i8  %a0,   ; %i0
                    i8  %a1,   ; %i1
                    i16 %a2,   ; %i2
                    i32 %a3,   ; %i3
                    i8* %a4,   ; %i4
                    i32 %a5,   ; %i5
                    i32 signext %a6,   ; [%fp+BIAS+176]
                    i8* %a7) { ; [%fp+BIAS+184]
  store volatile i8 %a0, i8* %a4
  store volatile i8 %a1, i8* %a4
  %p16 = bitcast i8* %a4 to i16*
  store volatile i16 %a2, i16* %p16
  %p32 = bitcast i8* %a4 to i32*
  store volatile i32 %a3, i32* %p32
  %pp = bitcast i8* %a4 to i8**
  store volatile i8* %a4, i8** %pp
  store volatile i32 %a5, i32* %p32
  store volatile i32 %a6, i32* %p32
  store volatile i8* %a7, i8** %pp
  ret void
}

; CHECK-LABEL: call_intarg:
; 16 saved + 8 args.
; CHECK: save %sp, -192, %sp
; Sign-extend and store the full 64 bits.
; CHECK: sra %i0, 0, [[R:%[gilo][0-7]]]
; Use %o0-%o5 for outgoing arguments
; CHECK: mov 5, %o5
; CHECK: stx [[R]], [%sp+2223]
; CHECK: call intarg
; CHECK-NOT: add %sp
; CHECK: restore
define void @call_intarg(i32 %i0, i8* %i1) {
  call void @intarg(i8 0, i8 1, i16 2, i32 3, i8* undef, i32 5, i32 %i0, i8* %i1)
  ret void
}

; CHECK-LABEL: floatarg:
; HARD: save %sp, -128, %sp
; HARD: ld [%fp+2307], [[F:%f[0-9]+]]
; HARD: fstod %f1,
; HARD: faddd %f2,
; HARD: faddd %f4,
; HARD: faddd %f6,
; HARD: fadds %f31, [[F]]
; SOFT: save %sp, -176, %sp
; SOFT: srl %i0, 0, %o0
; SOFT-NEXT: call __extendsfdf2
; SOFT: mov  %o0, %i0
; SOFT: mov  %i1, %o0
; SOFT: mov  %i2, %o0
; SOFT: mov  %i3, %o0
; SOFT: ld [%fp+2299], %o0
; SOFT: ld [%fp+2307], %o1
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

; CHECK-LABEL: call_floatarg:
; CHECK: save %sp, -272, %sp
; Store 8 bytes in full slot.
; HARD: std %f2, [%sp+2311]
; Store 4 bytes, right-aligned in slot.
; HARD: st %f1, [%sp+2307]
; HARD: fmovd %f2, %f4
; SOFT: stx %i1, [%sp+2311]
; SOFT: stx %i0, [%sp+2303]
; SOFT: stx %i2, [%sp+2295]
; SOFT: stx %i2, [%sp+2287]
; SOFT: stx %i2, [%sp+2279]
; SOFT: stx %i2, [%sp+2271]
; SOFT: stx %i2, [%sp+2263]
; SOFT: stx %i2, [%sp+2255]
; SOFT: stx %i2, [%sp+2247]
; SOFT: stx %i2, [%sp+2239]
; SOFT: stx %i2, [%sp+2231]
; SOFT: stx %i2, [%sp+2223]
; SOFT: mov  %i2, %o0
; SOFT: mov  %i1, %o1
; SOFT: mov  %i1, %o2
; SOFT: mov  %i1, %o3
; SOFT: mov  %i2, %o4
; SOFT: mov  %i2, %o5
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

; CHECK-LABEL: mixedarg:
; CHECK: ldx [%fp+2247]
; CHECK: ldx [%fp+2231]
; SOFT: ldx [%fp+2239], %i0
; HARD: fstod %f3
; HARD: faddd %f6
; HARD: faddd %f16
; SOFT: mov  %o0, %i1
; SOFT-NEXT: mov  %i3, %o0
; SOFT-NEXT: mov  %i1, %o1
; SOFT-NEXT: call __adddf3
; SOFT: mov  %o0, %i1
; SOFT-NEXT: mov  %i0, %o0
; SOFT-NEXT: mov  %i1, %o1
; SOFT-NEXT: call __adddf3
; HARD: std %f0, [%i1]
; SOFT: stx %o0, [%i5]

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

; CHECK-LABEL: call_mixedarg:
; CHECK: stx %i2, [%sp+2247]
; SOFT:  stx %i1, [%sp+2239]
; CHECK: stx %i0, [%sp+2223]
; HARD: fmovd %f2, %f6
; HARD: fmovd %f2, %f16
; SOFT: mov  %i1, %o3
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
; CHECK-LABEL: inreg_fi:
; SOFT: srlx %i0, 32, [[R:%[gilo][0-7]]]
; HARD: fstoi %f1
; SOFT: call __fixsfsi
; HARD: srlx %i0, 32, [[R:%[gilo][0-7]]]
; CHECK: sub [[R]],
define i32 @inreg_fi(i32 inreg %a0,     ; high bits of %i0
                     float inreg %a1) { ; %f1
  %b1 = fptosi float %a1 to i32
  %rv = sub i32 %a0, %b1
  ret i32 %rv
}

; CHECK-LABEL: call_inreg_fi:
; Allocate space for 6 arguments, even when only 2 are used.
; CHECK: save %sp, -176, %sp
; HARD:  sllx %i1, 32, %o0
; HARD:  fmovs %f5, %f1
; SOFT:  srl %i2, 0, %i0
; SOFT:  sllx %i1, 32, %i1
; SOFT:  or %i1, %i0, %o0
; CHECK: call inreg_fi
define void @call_inreg_fi(i32* %p, i32 %i1, float %f5) {
  %x = call i32 @inreg_fi(i32 %i1, float %f5)
  ret void
}

; CHECK-LABEL: inreg_ff:
; HARD: fsubs %f0, %f1, %f0
; SOFT: srlx %i0, 32, %o0
; SOFT: srl %i0, 0, %o1
; SOFT: call __subsf3
define float @inreg_ff(float inreg %a0,   ; %f0
                       float inreg %a1) { ; %f1
  %rv = fsub float %a0, %a1
  ret float %rv
}

; CHECK-LABEL: call_inreg_ff:
; HARD: fmovs %f3, %f0
; HARD: fmovs %f5, %f1
; SOFT: srl %i2, 0, %i0
; SOFT: sllx %i1, 32, %i1
; SOFT: or %i1, %i0, %o0
; CHECK: call inreg_ff
define void @call_inreg_ff(i32* %p, float %f3, float %f5) {
  %x = call float @inreg_ff(float %f3, float %f5)
  ret void
}

; CHECK-LABEL: inreg_if:
; HARD: fstoi %f0
; SOFT: srlx %i0, 32, %o0
; SOFT: call __fixsfsi
; CHECK: sub %i0
define i32 @inreg_if(float inreg %a0, ; %f0
                     i32 inreg %a1) { ; low bits of %i0
  %b0 = fptosi float %a0 to i32
  %rv = sub i32 %a1, %b0
  ret i32 %rv
}

; CHECK-LABEL: call_inreg_if:
; HARD: fmovs %f3, %f0
; HARD: mov %i2, %o0
; SOFT: srl %i2, 0, %i0
; SOFT: sllx %i1, 32, %i1
; SOFT: or %i1, %i0, %o0
; CHECK: call inreg_if
define void @call_inreg_if(i32* %p, float %f3, i32 %i2) {
  %x = call i32 @inreg_if(float %f3, i32 %i2)
  ret void
}

; The frontend shouldn't do this. Just pass i64 instead.
; CHECK-LABEL: inreg_ii:
; CHECK: srlx %i0, 32, [[R:%[gilo][0-7]]]
; CHECK: sub %i0, [[R]], %i0
define i32 @inreg_ii(i32 inreg %a0,   ; high bits of %i0
                     i32 inreg %a1) { ; low bits of %i0
  %rv = sub i32 %a1, %a0
  ret i32 %rv
}

; CHECK-LABEL: call_inreg_ii:
; CHECK: srl %i2, 0, [[R2:%[gilo][0-7]]]
; CHECK: sllx %i1, 32, [[R1:%[gilo][0-7]]]
; CHECK: or [[R1]], [[R2]], %o0
; CHECK: call inreg_ii
define void @call_inreg_ii(i32* %p, i32 %i1, i32 %i2) {
  %x = call i32 @inreg_ii(i32 %i1, i32 %i2)
  ret void
}

; Structs up to 32 bytes in size can be returned in registers.
; CHECK-LABEL: ret_i64_pair:
; CHECK: ldx [%i2], %i0
; CHECK: ldx [%i3], %i1
define { i64, i64 } @ret_i64_pair(i32 %a0, i32 %a1, i64* %p, i64* %q) {
  %r1 = load i64, i64* %p
  %rv1 = insertvalue { i64, i64 } undef, i64 %r1, 0
  store i64 0, i64* %p
  %r2 = load i64, i64* %q
  %rv2 = insertvalue { i64, i64 } %rv1, i64 %r2, 1
  ret { i64, i64 } %rv2
}

; CHECK-LABEL: call_ret_i64_pair:
; CHECK: call ret_i64_pair
; CHECK: stx %o0, [%i0]
; CHECK: stx %o1, [%i0]
define void @call_ret_i64_pair(i64* %i0) {
  %rv = call { i64, i64 } @ret_i64_pair(i32 undef, i32 undef,
                                        i64* undef, i64* undef)
  %e0 = extractvalue { i64, i64 } %rv, 0
  store volatile i64 %e0, i64* %i0
  %e1 = extractvalue { i64, i64 } %rv, 1
  store i64 %e1, i64* %i0
  ret void
}

; This is not a C struct, the i32 member uses 8 bytes, but the float only 4.
; CHECK-LABEL: ret_i32_float_pair:
; CHECK: ld [%i2], %i0
; HARD: ld [%i3], %f2
; SOFT: ld [%i3], %i1
define { i32, float } @ret_i32_float_pair(i32 %a0, i32 %a1,
                                          i32* %p, float* %q) {
  %r1 = load i32, i32* %p
  %rv1 = insertvalue { i32, float } undef, i32 %r1, 0
  store i32 0, i32* %p
  %r2 = load float, float* %q
  %rv2 = insertvalue { i32, float } %rv1, float %r2, 1
  ret { i32, float } %rv2
}

; CHECK-LABEL: call_ret_i32_float_pair:
; CHECK: call ret_i32_float_pair
; CHECK: st %o0, [%i0]
; HARD: st %f2, [%i1]
; SOFT: st %o1, [%i1]
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
; CHECK-LABEL: ret_i32_float_packed:
; CHECK: ld [%i2], [[R:%[gilo][0-7]]]
; HARD: ld [%i3], %f1
; SOFT: ld [%i3], %i1
; CHECK: sllx [[R]], 32, %i0
define inreg { i32, float } @ret_i32_float_packed(i32 %a0, i32 %a1,
                                                  i32* %p, float* %q) {
  %r1 = load i32, i32* %p
  %rv1 = insertvalue { i32, float } undef, i32 %r1, 0
  store i32 0, i32* %p
  %r2 = load float, float* %q
  %rv2 = insertvalue { i32, float } %rv1, float %r2, 1
  ret { i32, float } %rv2
}

; CHECK-LABEL: call_ret_i32_float_packed:
; CHECK: call ret_i32_float_packed
; CHECK: srlx %o0, 32, [[R:%[gilo][0-7]]]
; CHECK: st [[R]], [%i0]
; HARD: st %f1, [%i1]
; SOFT: st %o0, [%i1]
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
; CHECK-LABEL: ret_i32_packed:
; CHECK: ld [%i2], [[R1:%[gilo][0-7]]]
; CHECK: ld [%i3], [[R2:%[gilo][0-7]]]
; CHECK: sllx [[R2]], 32, [[R3:%[gilo][0-7]]]
; CHECK: or [[R3]], [[R1]], %i0
define inreg { i32, i32 } @ret_i32_packed(i32 %a0, i32 %a1,
                                          i32* %p, i32* %q) {
  %r1 = load i32, i32* %p
  %rv1 = insertvalue { i32, i32 } undef, i32 %r1, 1
  store i32 0, i32* %p
  %r2 = load i32, i32* %q
  %rv2 = insertvalue { i32, i32 } %rv1, i32 %r2, 0
  ret { i32, i32 } %rv2
}

; CHECK-LABEL: call_ret_i32_packed:
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
; CHECK-LABEL: ret_sext:
; CHECK: sra %i0, 0, %i0
define signext i32 @ret_sext(i32 %a0) {
  ret i32 %a0
}

; CHECK-LABEL: ret_zext:
; CHECK: srl %i0, 0, %i0
define zeroext i32 @ret_zext(i32 %a0) {
  ret i32 %a0
}

; CHECK-LABEL: ret_nosext:
; CHECK-NOT: sra
define signext i32 @ret_nosext(i32 signext %a0) {
  ret i32 %a0
}

; CHECK-LABEL: ret_nozext:
; CHECK-NOT: srl
define signext i32 @ret_nozext(i32 signext %a0) {
  ret i32 %a0
}

; CHECK-LABEL: test_register_directive:
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

; CHECK-LABEL: test_large_stack:

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
  %buffer1.sub = getelementptr inbounds [16384 x i8], [16384 x i8]* %buffer1, i32 0, i32 0
  %0 = call i32 @use_buf(i32 16384, i8* %buffer1.sub)
  ret i32 %0
}

declare i32 @use_buf(i32, i8*)

; CHECK-LABEL: test_fp128_args:
; HARD-DAG:   std %f0, [%fp+{{.+}}]
; HARD-DAG:   std %f2, [%fp+{{.+}}]
; HARD-DAG:   std %f6, [%fp+{{.+}}]
; HARD-DAG:   std %f4, [%fp+{{.+}}]
; HARD:       add %fp, [[Offset:[0-9]+]], %o0
; HARD:       call _Qp_add
; HARD:       ldd [%fp+[[Offset]]], %f0
; SOFT-DAG:       mov  %i0, %o0
; SOFT-DAG:       mov  %i1, %o1
; SOFT-DAG:       mov  %i2, %o2
; SOFT-DAG:       mov  %i3, %o3
; SOFT:           call __addtf3
; SOFT:           mov  %o0, %i0
; SOFT:           mov  %o1, %i1

define fp128 @test_fp128_args(fp128 %a, fp128 %b) {
entry:
  %0 = fadd fp128 %a, %b
  ret fp128 %0
}

declare i64 @receive_fp128(i64 %a, ...)

; CHECK-LABEL: test_fp128_variable_args:
; HARD-DAG:   std %f4, [%sp+[[Offset0:[0-9]+]]]
; HARD-DAG:   std %f6, [%sp+[[Offset1:[0-9]+]]]
; HARD-DAG:   ldx [%sp+[[Offset0]]], %o2
; HARD-DAG:   ldx [%sp+[[Offset1]]], %o3
; SOFT-DAG:   mov  %i0, %o0
; SOFT-DAG:   mov  %i1, %o1
; SOFT-DAG:   mov  %i2, %o2
; CHECK:      call receive_fp128
define i64 @test_fp128_variable_args(i64 %a, fp128 %b) {
entry:
  %0 = call i64 (i64, ...) @receive_fp128(i64 %a, fp128 %b)
  ret i64 %0
}

; CHECK-LABEL: test_call_libfunc:
; HARD:   st %f1, [%fp+[[Offset0:[0-9]+]]]
; HARD:   fmovs %f3, %f1
; SOFT:   srl %i1, 0, %o0
; CHECK:  call cosf
; HARD:   st %f0, [%fp+[[Offset1:[0-9]+]]]
; HARD:   ld [%fp+[[Offset0]]], %f1
; SOFT:   mov  %o0, %i1
; SOFT:   srl %i0, 0, %o0
; CHECK:  call sinf
; HARD:   ld [%fp+[[Offset1]]], %f1
; HARD:   fmuls %f1, %f0, %f0
; SOFT:   mov  %o0, %i0
; SOFT:   mov  %i1, %o0
; SOFT:   mov  %i0, %o1
; SOFT:   call __mulsf3
; SOFT:   sllx %o0, 32, %i0

define inreg float @test_call_libfunc(float %arg0, float %arg1) {
entry:
  %0 = tail call inreg float @cosf(float %arg1)
  %1 = tail call inreg float @sinf(float %arg0)
  %2 = fmul float %0, %1
  ret float %2
}

declare inreg float @cosf(float %arg) readnone nounwind
declare inreg float @sinf(float %arg) readnone nounwind
