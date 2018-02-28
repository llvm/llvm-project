; RUN: llc < %s -march=sparc -disable-sparc-delay-filler -disable-sparc-leaf-proc | FileCheck %s --check-prefix=CHECK --check-prefix=HARD --check-prefix=CHECK-BE
; RUN: llc < %s -march=sparcel -disable-sparc-delay-filler -disable-sparc-leaf-proc | FileCheck %s --check-prefix=CHECK --check-prefix=HARD --check-prefix=CHECK-LE
; RUN: llc < %s -march=sparc -disable-sparc-delay-filler -disable-sparc-leaf-proc -mattr=soft-float | FileCheck %s --check-prefix=CHECK --check-prefix=SOFT --check-prefix=CHECK-BE

; CHECK-LABEL: intarg:
; The save/restore frame is not strictly necessary here, but we would need to
; refer to %o registers instead.
; CHECK: save %sp, -96, %sp
; CHECK: ld [%fp+96], [[R2:%[gilo][0-7]]]
; CHECK: ld [%fp+92], [[R1:%[gilo][0-7]]]
; CHECK: stb %i0, [%i4]
; CHECK: stb %i1, [%i4]
; CHECK: sth %i2, [%i4]
; CHECK: st  %i3, [%i4]
; CHECK: st  %i4, [%i4]
; CHECK: st  %i5, [%i4]
; CHECK: st  [[R1]], [%i4]
; CHECK: st  [[R2]], [%i4]
; CHECK: restore
define void @intarg(i8  %a0,   ; %i0
                    i8  %a1,   ; %i1
                    i16 %a2,   ; %i2
                    i32 %a3,   ; %i3
                    i8* %a4,   ; %i4
                    i32 %a5,   ; %i5
                    i32 signext %a6,   ; [%fp+92]
                    i8* %a7) { ; [%fp+96]
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
; CHECK: save %sp, -104, %sp
; Use %o0-%o5 for outgoing arguments
; CHECK: mov 5, %o5
; CHECK: st %i0, [%sp+92]
; CHECK: call intarg
; CHECK-NOT: add %sp
; CHECK: restore
define void @call_intarg(i32 %i0, i8* %i1) {
  call void @intarg(i8 0, i8 1, i16 2, i32 3, i8* undef, i32 5, i32 %i0, i8* %i1)
  ret void
}

;; Verify doubles starting with an even reg, starting with an odd reg,
;; straddling the boundary of regs and mem, and floats in regs and mem.
;
; CHECK-LABEL: floatarg:
; HARD: save %sp, -120, %sp
; HARD: mov %i5, %g2
; HARD-NEXT: ld [%fp+92], %g3
; HARD-NEXT: mov %i4, %i5
; HARD-NEXT: ! kill
; HARD-NEXT: std %g2, [%fp+-24]
; HARD-NEXT: mov %i3, %i4
; HARD-NEXT: std %i4, [%fp+-16]
; HARD-NEXT: ! kill
; HARD-NEXT: std %i0, [%fp+-8]
; HARD-NEXT: st %i2, [%fp+-28]
; HARD-NEXT: ld [%fp+104], %f0
; HARD-NEXT: ldd [%fp+96], %f2
; HARD-NEXT: ld [%fp+-28], %f1
; HARD-NEXT: ldd [%fp+-8], %f4
; HARD-NEXT: ldd [%fp+-16], %f6
; HARD-NEXT: ldd [%fp+-24], %f8
; HARD-NEXT: fstod %f1, %f10
; HARD-NEXT: faddd %f4, %f10, %f4
; HARD-NEXT: faddd %f6, %f4, %f4
; HARD-NEXT: faddd %f8, %f4, %f4
; HARD-NEXT: faddd %f2, %f4, %f2
; HARD-NEXT: fstod %f0, %f0
; HARD-NEXT: faddd %f0, %f2, %f0
; SOFT: save %sp, -96, %sp
; SOFT: ld [%fp+104], %l0
; SOFT-NEXT: ld [%fp+96], %l1
; SOFT-NEXT: ld [%fp+100], %l2
; SOFT-NEXT: ld [%fp+92], %l3
; SOFT-NEXT:  mov  %i2, %o0
; SOFT-NEXT: call __extendsfdf2
; SOFT-NEXT: nop
; SOFT-NEXT:  mov  %o0, %o2
; SOFT-NEXT:  mov  %o1, %o3
; SOFT-NEXT:  mov  %i0, %o0
; SOFT-NEXT:  mov  %i1, %o1
; SOFT-NEXT:  call __adddf3
; SOFT-NEXT:  nop
; SOFT-NEXT:  mov  %o0, %o2
; SOFT-NEXT:  mov  %o1, %o3
; SOFT-NEXT:  mov  %i3, %o0
; SOFT-NEXT:  mov  %i4, %o1
; SOFT-NEXT:  call __adddf3
; SOFT-NEXT:  nop
; SOFT-NEXT:  mov  %o0, %o2
; SOFT-NEXT:  mov  %o1, %o3
; SOFT-NEXT:  mov  %i5, %o0
; SOFT-NEXT:  mov  %l3, %o1
; SOFT-NEXT:  call __adddf3
; SOFT-NEXT:  nop
; SOFT-NEXT:  mov  %o0, %o2
; SOFT-NEXT:  mov  %o1, %o3
; SOFT-NEXT:  mov  %l1, %o0
; SOFT-NEXT:  mov  %l2, %o1
; SOFT-NEXT:  call __adddf3
; SOFT-NEXT:  nop
; SOFT-NEXT:  mov  %o0, %i0
; SOFT-NEXT:  mov  %o1, %i1
; SOFT-NEXT:  mov  %l0, %o0
; SOFT-NEXT:  call __extendsfdf2
; SOFT-NEXT:  nop
; SOFT-NEXT:  mov  %i0, %o2
; SOFT-NEXT:  mov  %i1, %o3
; SOFT-NEXT:  call __adddf3
; SOFT-NEXT:  nop
; SOFT-NEXT:  mov  %o0, %i0
; SOFT-NEXT:  mov  %o1, %i1
; CHECK: restore
define double @floatarg(double %a0,   ; %i0,%i1
                        float %a1,    ; %i2
                        double %a2,   ; %i3, %i4
                        double %a3,   ; %i5, [%fp+92] (using 4 bytes)
                        double %a4,   ; [%fp+96] (using 8 bytes)
                        float %a5) {  ; [%fp+104] (using 4 bytes)
  %d1 = fpext float %a1 to double
  %s1 = fadd double %a0, %d1
  %s2 = fadd double %a2, %s1
  %s3 = fadd double %a3, %s2
  %s4 = fadd double %a4, %s3
  %d5 = fpext float %a5 to double
  %s5 = fadd double %d5, %s4
  ret double %s5
}

; CHECK-LABEL: call_floatarg:
; HARD: save %sp, -112, %sp
; HARD: mov %i2, %o1
; HARD-NEXT: mov %i1, %o0
; HARD-NEXT: st %i0, [%sp+104]
; HARD-NEXT: std %o0, [%sp+96]
; HARD-NEXT: st %o1, [%sp+92]
; HARD-NEXT: mov %i0, %o2
; HARD-NEXT: mov %i1, %o3
; HARD-NEXT: mov %o1, %o4
; HARD-NEXT: mov %i1, %o5
; HARD-NEXT: call floatarg
; HARD: std %f0, [%i4]
; SOFT: st %i0, [%sp+104]
; SOFT-NEXT:  st %i2, [%sp+100]
; SOFT-NEXT:  st %i1, [%sp+96]
; SOFT-NEXT:  st %i2, [%sp+92]
; SOFT-NEXT:  mov  %i1, %o0
; SOFT-NEXT:  mov  %i2, %o1
; SOFT-NEXT:  mov  %i0, %o2
; SOFT-NEXT:  mov  %i1, %o3
; SOFT-NEXT:  mov  %i2, %o4
; SOFT-NEXT:  mov  %i1, %o5
; SOFT-NEXT:  call floatarg
; SOFT:  std %o0, [%i4]
; CHECK: restore
define void @call_floatarg(float %f1, double %d2, float %f5, double *%p) {
  %r = call double @floatarg(double %d2, float %f1, double %d2, double %d2,
                             double %d2, float %f1)
  store double %r, double* %p
  ret void
}

;; i64 arguments should effectively work the same as double: split
;; into two locations.  This is different for little-endian vs big
;; endian, since the 64-bit math needs to be split
; CHECK-LABEL: i64arg:
; CHECK:  save %sp, -96, %sp
; CHECK-BE: ld [%fp+100], %g2
; CHECK-BE-NEXT: ld [%fp+96], %g3
; CHECK-BE-NEXT: ld [%fp+92], %g4
; CHECK-BE-NEXT: addcc %i1, %i2, %i1
; CHECK-BE-NEXT: addxcc %i0, 0, %i0
; CHECK-BE-NEXT: addcc %i4, %i1, %i1
; CHECK-BE-NEXT: addxcc %i3, %i0, %i0
; CHECK-BE-NEXT: addcc %g4, %i1, %i1
; CHECK-BE-NEXT: ld [%fp+104], %i2
; CHECK-BE-NEXT: addxcc %i5, %i0, %i0
; CHECK-BE-NEXT: addcc %g2, %i1, %i1
; CHECK-BE-NEXT: addxcc %g3, %i0, %i0
; CHECK-BE-NEXT: addcc %i2, %i1, %i1
; CHECK-BE-NEXT: addxcc %i0, 0, %i0
;
; CHECK-LE: ld [%fp+96], %g2
; CHECK-LE-NEXT: ld [%fp+100], %g3
; CHECK-LE-NEXT: ld [%fp+92], %g4
; CHECK-LE-NEXT: addcc %i0, %i2, %i0
; CHECK-LE-NEXT: addxcc %i1, 0, %i1
; CHECK-LE-NEXT: addcc %i3, %i0, %i0
; CHECK-LE-NEXT: addxcc %i4, %i1, %i1
; CHECK-LE-NEXT: addcc %i5, %i0, %i0
; CHECK-LE-NEXT: ld [%fp+104], %i2
; CHECK-LE-NEXT: addxcc %g4, %i1, %i1
; CHECK-LE-NEXT: addcc %g2, %i0, %i0
; CHECK-LE-NEXT: addxcc %g3, %i1, %i1
; CHECK-LE-NEXT: addcc %i2, %i0, %i0
; CHECK-LE-NEXT: addxcc %i1, 0, %i1
; CHECK-NEXT: restore


define i64 @i64arg(i64 %a0,    ; %i0,%i1
		   i32 %a1,    ; %i2
		   i64 %a2,    ; %i3, %i4
		   i64 %a3,    ; %i5, [%fp+92] (using 4 bytes)
		   i64 %a4,    ; [%fp+96] (using 8 bytes)
                   i32 %a5) {  ; [%fp+104] (using 4 bytes)
  %a1L = zext i32 %a1 to i64
  %s1 = add i64 %a0, %a1L
  %s2 = add i64 %a2, %s1
  %s3 = add i64 %a3, %s2
  %s4 = add i64 %a4, %s3
  %a5L = zext i32 %a5 to i64
  %s5 = add i64 %a5L, %s4
  ret i64 %s5
}

; CHECK-LABEL: call_i64arg:
; CHECK: save %sp, -112, %sp
; CHECK: st %i0, [%sp+104]
; CHECK-NEXT: st %i2, [%sp+100]
; CHECK-NEXT: st %i1, [%sp+96]
; CHECK-NEXT: st %i2, [%sp+92]
; CHECK-NEXT: mov      %i1, %o0
; CHECK-NEXT: mov      %i2, %o1
; CHECK-NEXT: mov      %i0, %o2
; CHECK-NEXT: mov      %i1, %o3
; CHECK-NEXT: mov      %i2, %o4
; CHECK-NEXT: mov      %i1, %o5
; CHECK-NEXT: call i64arg
; CHECK: std %o0, [%i3]
; CHECK-NEXT: restore

define void @call_i64arg(i32 %a0, i64 %a1, i64* %p) {
  %r = call i64 @i64arg(i64 %a1, i32 %a0, i64 %a1, i64 %a1, i64 %a1, i32 %a0)
  store i64 %r, i64* %p
  ret void
}
