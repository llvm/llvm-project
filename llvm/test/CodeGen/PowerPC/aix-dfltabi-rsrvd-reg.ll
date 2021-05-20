;; Test to ensure that we are not using any of the aliased reserved registers
;; under the Extended Altivec ABI on AIX.
; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mattr=+altivec \
; RUN:     -stop-after=machine-cp -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN:   FileCheck %s --check-prefix=DFLABI
; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mattr=+altivec -vec-extabi\
; RUN:     -stop-after=machine-cp -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN:   FileCheck %s --check-prefix=EXTABI

define double @dbl_test(double %a, double* %b) local_unnamed_addr {
entry:
  %0 = load volatile double, double* %b, align 4
  %add = fadd double %0, %a
  store volatile double %add, double* %b, align 4
  ;; Clobbered all vector and floating point registers. In the default Altivec
  ;; ABI this forces a register spill since no registers are free to use.
  tail call void asm sideeffect "nop", "~{v19},~{v18},~{v17},~{v16},~{v15},~{v14},~{v13},~{v12},~{v11},~{v10},~{v9},~{v8},~{v7},~{v6},~{v5},~{v4},~{v3},~{v2},~{v1},~{v0},~{f0},~{f1},~{f2},~{f3},~{f4},~{f5},~{f6},~{f7},~{f8},~{f9},~{f10},~{f11},~{f12},~{f13},~{f14},~{f15},~{f16},~{f17},~{f18},~{f19},~{f20},~{f21},~{f22},~{f23},~{f24},~{f25},~{f26},~{f27},~{f28},~{f29},~{f30},~{f31}"()
  %mul = fmul double %a, %a
  %1 = load volatile double, double* %b, align 4
  %add1 = fadd double %mul, %1
  store volatile double %add1, double* %b, align 4
  %2 = load volatile double, double* %b, align 4
  ret double %2
}

define <4 x i32> @vec_test(<4 x i32> %a,  <4 x i32>* %b) local_unnamed_addr {
entry:
  %0 = load volatile <4 x i32>, <4 x i32>* %b, align 4
  %add = add <4 x i32> %0, %a
  store volatile <4 x i32> %add, <4 x i32>* %b, align 4
  tail call void asm sideeffect "nop", "~{v19},~{v18},~{v17},~{v16},~{v15},~{v14},~{v13},~{v12},~{v11},~{v10},~{v9},~{v8},~{v7},~{v6},~{v5},~{v4},~{v3},~{v2},~{v1},~{v0},~{f0},~{f1},~{f2},~{f3},~{f4},~{f5},~{f6},~{f7},~{f8},~{f9},~{f10},~{f11},~{f12},~{f13},~{f14},~{f15},~{f16},~{f17},~{f18},~{f19},~{f20},~{f21},~{f22},~{f23},~{f24},~{f25},~{f26},~{f27},~{f28},~{f29},~{f30},~{f31}"()
  %mul = mul <4 x i32> %a, %a
  %1 = load volatile <4 x i32>, <4 x i32>* %b, align 4
  %add1 = add <4 x i32> %mul, %1
  store volatile <4 x i32> %add1, <4 x i32>* %b, align 4
  %2 = load volatile <4 x i32>, <4 x i32>* %b, align 4
  ret <4 x i32> %2
}

; DFLABI-LABEL:   dbl_test

; DFLABI-NOT:     $v20
; DFLABI-NOT:     $v21
; DFLABI-NOT:     $v22
; DFLABI-NOT:     $v23
; DFLABI-NOT:     $v24
; DFLABI-NOT:     $v25
; DFLABI-NOT:     $v26
; DFLABI-NOT:     $v27
; DFLABI-NOT:     $v28
; DFLABI-NOT:     $v29
; DFLABI-NOT:     $v30
; DFLABI-NOT:     $v31

; DFLABI-NOT:     $vf20
; DFLABI-NOT:     $vf21
; DFLABI-NOT:     $vf22
; DFLABI-NOT:     $vf23
; DFLABI-NOT:     $vf24
; DFLABI-NOT:     $vf25
; DFLABI-NOT:     $vf26
; DFLABI-NOT:     $vf27
; DFLABI-NOT:     $vf28
; DFLABI-NOT:     $vf29
; DFLABI-NOT:     $vf30
; DFLABI-NOT:     $vf31

; DFLABI-NOT:     $vs20
; DFLABI-NOT:     $vs21
; DFLABI-NOT:     $vs22
; DFLABI-NOT:     $vs23
; DFLABI-NOT:     $vs24
; DFLABI-NOT:     $vs25
; DFLABI-NOT:     $vs26
; DFLABI-NOT:     $vs27
; DFLABI-NOT:     $vs28
; DFLABI-NOT:     $vs29
; DFLABI-NOT:     $vs30
; DFLABI-NOT:     $vs31

; EXTABI-LABEL:   vec_test
; EXTABI:         liveins:
; EXTABI-NEXT:     - { reg: '$f1', virtual-reg: '' }
; EXTABI-NEXT:     - { reg: '$x4', virtual-reg: '' }
; EXTABI:         body:             |
; EXTABI:         bb.0.entry:
; EXTABI:         liveins: $f1, $x4
; EXTABI-DAG:     renamable $f0 = XFLOADf64 $zero8, renamable $x4 :: (volatile load (s64) from %ir.b, align 4)
; EXTABI-DAG:     renamable $f0 = nofpexcept XSADDDP killed renamable $f0, renamable $f1, implicit $rm
; EXTABI-DAG:     renamable $vf31 = nofpexcept XSMULDP killed renamable $f1, renamable $f1, implicit $rm
; EXTABI:         XFSTOREf64 killed renamable $f0, $zero8, renamable $x4 :: (volatile store (s64) into %ir.b, align 4)
; EXTABI-LABEL:   INLINEASM
; EXTABI-DAG:     renamable $f0 = XFLOADf64 $zero8, renamable $x4 :: (volatile load (s64) from %ir.b, align 4)
; EXTABI-DAG:     renamable $f0 = nofpexcept XSADDDP killed renamable $vf31, killed renamable $f0, implicit $rm
; EXTABI-DAG:     XFSTOREf64 killed renamable $f0, $zero8, renamable $x4 :: (volatile store (s64) into %ir.b, align 4)
; EXTABI:         renamable $f1 = XFLOADf64 $zero8, killed renamable $x4 :: (volatile load (s64) from %ir.b, align 4)

; DFLABI-LABEL:   vec_test

; DFLABI-NOT:     $v20
; DFLABI-NOT:     $v21
; DFLABI-NOT:     $v22
; DFLABI-NOT:     $v23
; DFLABI-NOT:     $v24
; DFLABI-NOT:     $v25
; DFLABI-NOT:     $v26
; DFLABI-NOT:     $v27
; DFLABI-NOT:     $v28
; DFLABI-NOT:     $v29
; DFLABI-NOT:     $v30
; DFLABI-NOT:     $v31

; DFLABI-NOT:     $vf20
; DFLABI-NOT:     $vf21
; DFLABI-NOT:     $vf22
; DFLABI-NOT:     $vf23
; DFLABI-NOT:     $vf24
; DFLABI-NOT:     $vf25
; DFLABI-NOT:     $vf26
; DFLABI-NOT:     $vf27
; DFLABI-NOT:     $vf28
; DFLABI-NOT:     $vf29
; DFLABI-NOT:     $vf30
; DFLABI-NOT:     $vf31

; DFLABI-NOT:     $vs20
; DFLABI-NOT:     $vs21
; DFLABI-NOT:     $vs22
; DFLABI-NOT:     $vs23
; DFLABI-NOT:     $vs24
; DFLABI-NOT:     $vs25
; DFLABI-NOT:     $vs26
; DFLABI-NOT:     $vs27
; DFLABI-NOT:     $vs28
; DFLABI-NOT:     $vs29
; DFLABI-NOT:     $vs30
; DFLABI-NOT:     $vs31

; EXTABI-LABEL:   vec_test

; EXTABI:         liveins:
; EXTABI-NEXT:     - { reg: '$v2', virtual-reg: '' }
; EXTABI-NEXT:     - { reg: '$x3', virtual-reg: '' }
; EXTABI:         body:             |
; EXTABI-DAG:     bb.0.entry:
; EXTABI-DAG:     liveins: $v2, $x3
; EXTABI-DAG:     renamable $v3 = LXVW4X $zero8, renamable $x3 :: (volatile load (s128) from %ir.b, align 4)
; EXTABI-DAG:     renamable $v31 = COPY $v2
; EXTABI-DAG:     renamable $v2 = VADDUWM killed renamable $v3, $v2
; EXTABI-LABEL:   INLINEASM    
; EXTABI-DAG:     renamable $v2 = LXVW4X $zero8, renamable $x3 :: (volatile load (s128) from %ir.b, align 4)
; EXTABI-DAG:     renamable $v3 = VMULUWM killed renamable $v31, renamable $v31
; EXTABI-DAG:     renamable $v2 = VADDUWM killed renamable $v3, killed renamable $v2
; EXTABI-DAG:     STXVW4X killed renamable $v2, $zero8, renamable $x3 :: (volatile store (s128) into %ir.b, align 4)
; EXTABI:         renamable $v2 = LXVW4X $zero8, killed renamable $x3 :: (volatile load (s128) from %ir.b, align 4)
