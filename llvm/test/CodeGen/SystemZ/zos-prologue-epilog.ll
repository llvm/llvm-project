; Test the generated function prologs/epilogs under XPLINK64 on z/OS
;
; RUN: llc < %s -mtriple=s390x-ibm-zos -mcpu=z13 | FileCheck --check-prefixes=CHECK64,CHECK %s

; Test prolog/epilog for non-XPLEAF.

; Small stack frame.
; CHECK-LABEL: func0
; CHECK64: stmg  6, 7, 1872(4)
; stmg instruction's displacement field must be 2064-dsa_size
; as per ABI
; CHECK64: aghi  4, -192

; CHECK64: lg  7, 2072(4)
; CHECK64: aghi  4, 192
; CHECK64: b 2(7)
define void @func0() {
  call i64 (i64) @fun(i64 10) 
  ret void
}

; Spill all GPR CSRs
; CHECK-LABEL: func1
; CHECK64: stmg 6, 15, 1904(4)
; CHECK64: aghi  4, -160

; CHECK64: lmg 7, 15, 2072(4)
; CHECK64: aghi  4, 160
; CHECK64: b 2(7)
define void @func1(i64 *%ptr) {
  %l01 = load volatile i64, i64 *%ptr
  %l02 = load volatile i64, i64 *%ptr
  %l03 = load volatile i64, i64 *%ptr
  %l04 = load volatile i64, i64 *%ptr
  %l05 = load volatile i64, i64 *%ptr
  %l06 = load volatile i64, i64 *%ptr
  %l07 = load volatile i64, i64 *%ptr
  %l08 = load volatile i64, i64 *%ptr
  %l09 = load volatile i64, i64 *%ptr
  %l10 = load volatile i64, i64 *%ptr
  %l11 = load volatile i64, i64 *%ptr
  %l12 = load volatile i64, i64 *%ptr
  %l13 = load volatile i64, i64 *%ptr
  %l14 = load volatile i64, i64 *%ptr
  %l15 = load volatile i64, i64 *%ptr
  %add01 = add i64 %l01, %l01
  %add02 = add i64 %l02, %add01
  %add03 = add i64 %l03, %add02
  %add04 = add i64 %l04, %add03
  %add05 = add i64 %l05, %add04
  %add06 = add i64 %l06, %add05
  %add07 = add i64 %l07, %add06
  %add08 = add i64 %l08, %add07
  %add09 = add i64 %l09, %add08
  %add10 = add i64 %l10, %add09
  %add11 = add i64 %l11, %add10
  %add12 = add i64 %l12, %add11
  %add13 = add i64 %l13, %add12
  %add14 = add i64 %l14, %add13
  %add15 = add i64 %l15, %add14
  store volatile i64 %add01, i64 *%ptr
  store volatile i64 %add02, i64 *%ptr
  store volatile i64 %add03, i64 *%ptr
  store volatile i64 %add04, i64 *%ptr
  store volatile i64 %add05, i64 *%ptr
  store volatile i64 %add06, i64 *%ptr
  store volatile i64 %add07, i64 *%ptr
  store volatile i64 %add08, i64 *%ptr
  store volatile i64 %add09, i64 *%ptr
  store volatile i64 %add10, i64 *%ptr
  store volatile i64 %add11, i64 *%ptr
  store volatile i64 %add12, i64 *%ptr
  store volatile i64 %add13, i64 *%ptr
  store volatile i64 %add14, i64 *%ptr
  store volatile i64 %add15, i64 *%ptr
  ret void
}


; Spill all FPRs and VRs
; CHECK-LABEL: func2
; CHECK64: stmg	6, 7, 1744(4)
; CHECK64: aghi  4, -320 
; CHECK64: std	15, {{[0-9]+}}(4)                      * 8-byte Folded Spill
; CHECK64: std	14, {{[0-9]+}}(4)                      * 8-byte Folded Spill
; CHECK64: std	13, {{[0-9]+}}(4)                      * 8-byte Folded Spill
; CHECK64: std	12, {{[0-9]+}}(4)                      * 8-byte Folded Spill
; CHECK64: std	11, {{[0-9]+}}(4)                      * 8-byte Folded Spill
; CHECK64: std	10, {{[0-9]+}}(4)                      * 8-byte Folded Spill
; CHECK64: std	9, {{[0-9]+}}(4)                       * 8-byte Folded Spill
; CHECK64: std	8, {{[0-9]+}}(4)                       * 8-byte Folded Spill
; CHECK64: vst	23, {{[0-9]+}}(4), 4                   * 16-byte Folded Spill
; CHECK64: vst	22, {{[0-9]+}}(4), 4                   * 16-byte Folded Spill
; CHECK64: vst	21, {{[0-9]+}}(4), 4                   * 16-byte Folded Spill
; CHECK64: vst	20, {{[0-9]+}}(4), 4                   * 16-byte Folded Spill
; CHECK64: vst	19, {{[0-9]+}}(4), 4                   * 16-byte Folded Spill
; CHECK64: vst	18, {{[0-9]+}}(4), 4                   * 16-byte Folded Spill
; CHECK64: vst	17, {{[0-9]+}}(4), 4                   * 16-byte Folded Spill
; CHECK64: vst	16, {{[0-9]+}}(4), 4                   * 16-byte Folded Spill

; CHECK64: ld	15, {{[0-9]+}}(4)                      * 8-byte Folded Reload
; CHECK64: ld	14, {{[0-9]+}}(4)                      * 8-byte Folded Reload
; CHECK64: ld	13, {{[0-9]+}}(4)                      * 8-byte Folded Reload
; CHECK64: ld	12, {{[0-9]+}}(4)                      * 8-byte Folded Reload
; CHECK64: ld	11, {{[0-9]+}}(4)                      * 8-byte Folded Reload
; CHECK64: ld	10, {{[0-9]+}}(4)                      * 8-byte Folded Reload
; CHECK64: ld	9, {{[0-9]+}}(4)                       * 8-byte Folded Reload
; CHECK64: ld	8, {{[0-9]+}}(4)                       * 8-byte Folded Reload
; CHECK64: vl	23, {{[0-9]+}}(4), 4                   * 16-byte Folded Reload
; CHECK64: vl	22, {{[0-9]+}}(4), 4                   * 16-byte Folded Reload
; CHECK64: vl	21, {{[0-9]+}}(4), 4                   * 16-byte Folded Reload
; CHECK64: vl	20, {{[0-9]+}}(4), 4                   * 16-byte Folded Reload
; CHECK64: vl	19, {{[0-9]+}}(4), 4                   * 16-byte Folded Reload
; CHECK64: vl	18, {{[0-9]+}}(4), 4                   * 16-byte Folded Reload
; CHECK64: vl	17, {{[0-9]+}}(4), 4                   * 16-byte Folded Reload
; CHECK64: vl	16, {{[0-9]+}}(4), 4                   * 16-byte Folded Reload
; CHECK64: lg  7, 2072(4)
; CHECK64: aghi  4, 320
; CHECK64: b 2(7)

define void @func2(double *%ptr, <2 x i64> *%vec_ptr) {
  %l00 = load volatile double, double *%ptr
  %l01 = load volatile double, double *%ptr
  %l02 = load volatile double, double *%ptr
  %l03 = load volatile double, double *%ptr
  %l04 = load volatile double, double *%ptr
  %l05 = load volatile double, double *%ptr
  %l06 = load volatile double, double *%ptr
  %l07 = load volatile double, double *%ptr
  %l08 = load volatile double, double *%ptr
  %l09 = load volatile double, double *%ptr
  %l10 = load volatile double, double *%ptr
  %l11 = load volatile double, double *%ptr
  %l12 = load volatile double, double *%ptr
  %l13 = load volatile double, double *%ptr
  %l14 = load volatile double, double *%ptr
  %l15 = load volatile double, double *%ptr
  %add00 = fadd double %l01, %l00
  %add01 = fadd double %l01, %add00
  %add02 = fadd double %l02, %add01
  %add03 = fadd double %l03, %add02
  %add04 = fadd double %l04, %add03
  %add05 = fadd double %l05, %add04
  %add06 = fadd double %l06, %add05
  %add07 = fadd double %l07, %add06
  %add08 = fadd double %l08, %add07
  %add09 = fadd double %l09, %add08
  %add10 = fadd double %l10, %add09
  %add11 = fadd double %l11, %add10
  %add12 = fadd double %l12, %add11
  %add13 = fadd double %l13, %add12
  %add14 = fadd double %l14, %add13
  %add15 = fadd double %l15, %add14
  store volatile double %add00, double *%ptr
  store volatile double %add01, double *%ptr
  store volatile double %add02, double *%ptr
  store volatile double %add03, double *%ptr
  store volatile double %add04, double *%ptr
  store volatile double %add05, double *%ptr
  store volatile double %add06, double *%ptr
  store volatile double %add07, double *%ptr
  store volatile double %add08, double *%ptr
  store volatile double %add09, double *%ptr
  store volatile double %add10, double *%ptr
  store volatile double %add11, double *%ptr
  store volatile double %add12, double *%ptr
  store volatile double %add13, double *%ptr
  store volatile double %add14, double *%ptr
  store volatile double %add15, double *%ptr

  %v00 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v01 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v02 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v03 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v04 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v05 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v06 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v07 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v08 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v09 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v10 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v11 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v12 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v13 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v14 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v15 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v16 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v17 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v18 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v19 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v20 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v21 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v22 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v23 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v24 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v25 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v26 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v27 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v28 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v29 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v30 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %v31 = load volatile <2 x i64>, <2 x i64> *%vec_ptr
  %vadd00 = add <2 x i64> %v00, %v00
  %vadd01 = add <2 x i64> %v01, %vadd00
  %vadd02 = add <2 x i64> %v02, %vadd01
  %vadd03 = add <2 x i64> %v03, %vadd02
  %vadd04 = add <2 x i64> %v04, %vadd03
  %vadd05 = add <2 x i64> %v05, %vadd04
  %vadd06 = add <2 x i64> %v06, %vadd05
  %vadd07 = add <2 x i64> %v07, %vadd06
  %vadd08 = add <2 x i64> %v08, %vadd07
  %vadd09 = add <2 x i64> %v09, %vadd08
  %vadd10 = add <2 x i64> %v10, %vadd09
  %vadd11 = add <2 x i64> %v11, %vadd10
  %vadd12 = add <2 x i64> %v12, %vadd11
  %vadd13 = add <2 x i64> %v13, %vadd12
  %vadd14 = add <2 x i64> %v14, %vadd13
  %vadd15 = add <2 x i64> %v15, %vadd14
  %vadd16 = add <2 x i64> %v16, %vadd15
  %vadd17 = add <2 x i64> %v17, %vadd16
  %vadd18 = add <2 x i64> %v18, %vadd17
  %vadd19 = add <2 x i64> %v19, %vadd18
  %vadd20 = add <2 x i64> %v20, %vadd19
  %vadd21 = add <2 x i64> %v21, %vadd20
  %vadd22 = add <2 x i64> %v22, %vadd21
  %vadd23 = add <2 x i64> %v23, %vadd22
  %vadd24 = add <2 x i64> %v24, %vadd23
  %vadd25 = add <2 x i64> %v25, %vadd24
  %vadd26 = add <2 x i64> %v26, %vadd25
  %vadd27 = add <2 x i64> %v27, %vadd26
  %vadd28 = add <2 x i64> %v28, %vadd27
  %vadd29 = add <2 x i64> %v29, %vadd28
  %vadd30 = add <2 x i64> %v30, %vadd29
  %vadd31 = add <2 x i64> %v31, %vadd30
  store volatile <2 x i64> %vadd00, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd01, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd02, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd03, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd04, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd05, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd06, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd07, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd08, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd09, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd10, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd11, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd12, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd13, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd14, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd15, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd16, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd17, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd18, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd19, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd20, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd21, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd22, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd23, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd24, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd25, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd26, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd27, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd28, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd29, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd30, <2 x i64> *%vec_ptr
  store volatile <2 x i64> %vadd31, <2 x i64> *%vec_ptr
  ret void
}

; Big stack frame, force the use of agfi before stmg
; despite not requiring stack extension routine.
; CHECK64: agfi  4, -1040768
; CHECK64: stmg  6, 7, 2064(4)
; CHECK64: agfi  4, 1040768
define void @func3() {
  %arr = alloca [130070 x i64], align 8
  %ptr = bitcast [130070 x i64]* %arr to i8*
  call i64 (i8*) @fun1(i8* %ptr)
  ret void
}

; Requires the saving of r4 due to variable sized
; object in stack frame. (Eg: VLA) Sets up frame pointer in r8
; CHECK64: stmg  4, 9, 1856(4)
; CHECK64: aghi  4, -192
; CHECK64: lgr     8, 4
; TODO Will change to basr with ADA introduction.
; CHECK64: brasl   7, @@ALCAXP
; CHECK64-NEXT: bcr     0, 3
; CHECK64: lmg	4, 9, 2048(4)
define i64 @func4(i64 %n) {
  %vla = alloca i64, i64 %n, align 8
  %call = call i64 @fun2(i64 %n, i64* nonnull %vla, i64* nonnull %vla)
  ret i64 %call
}

; Require saving of r4 and in addition, a displacement large enough
; to force use of agfi before stmg.
; CHECK64: lgr	0, 4
; CHECK64: agfi	4, -1040192
; CHECK64: stmg  4, 9, 2048(4)
; CHECK64: lgr     8, 4
; TODO Will change to basr with ADA introduction.
; CHECK64: brasl   7, @@ALCAXP
; CHECK64-NEXT: bcr     0, 3
;; CHECK64: lmg 4, 9, 2048(4)
define i64 @func5(i64 %n) {
  %vla = alloca i64, i64 %n, align 8
  %arr = alloca [130000 x i64], align 8
  %ptr = bitcast [130000 x i64]* %arr to i64*
  %call = call i64 @fun2(i64 %n, i64* nonnull %vla, i64* %ptr)
  ret i64 %call
}

; CHECK-LABEL: large_stack
; CHECK64: agfi  4, -1048768
; CHECK64-NEXT: llgt  3, 1208
; CHECK64-NEXT: cg  4, 64(3)
; CHECK64-NEXT: jhe
; CHECK64: * %bb.1:
; CHECK64: lg  3, 72(3)
; CHECK64: basr  3, 3
; CHECK64: stmg  6, 7, 2064(4)
define void @large_stack() {
  %arr = alloca [131072 x i64], align 8
  %ptr = bitcast [131072 x i64]* %arr to i8*
  call i64 (i8*) @fun1(i8* %ptr)
  ret void
}

declare i64 @fun(i64 %arg0)
declare i64 @fun1(i8* %ptr)
declare i64 @fun2(i64 %n, i64* %arr0, i64* %arr1)
