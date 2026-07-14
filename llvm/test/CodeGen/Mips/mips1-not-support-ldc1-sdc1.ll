; RUN: llc -march=mipsel -mcpu=mips1 < %s | FileCheck %s -check-prefix=MIPS1-LE
; RUN: llc -march=mips -mcpu=mips1 < %s | FileCheck %s -check-prefix=MIPS1-BE

@test = global double 1.000000e+00, align 8

define double @test_lwc1() {
; MIPS1-LE-LABEL: test_lwc1:
; MIPS1-LE:       # %bb.0: # %entry
; MIPS1-LE-NEXT:    lui	$1, %hi(test)
; MIPS1-LE-NEXT:    lwc1 $f0, %lo(test)($1)
; MIPS1-LE-NEXT:    lwc1 $f1, %lo(test+4)($1)
; MIPS1-LE-NEXT:    jr $ra
; MIPS1-LE-NEXT:    nop

; MIPS1-BE-LABEL: test_lwc1:
; MIPS1-BE:       # %bb.0: # %entry
; MIPS1-BE-NEXT:    lui	$1, %hi(test)
; MIPS1-BE-NEXT:    lwc1 $f0, %lo(test+4)($1)
; MIPS1-BE-NEXT:    lwc1 $f1, %lo(test)($1)
; MIPS1-BE-NEXT:    jr $ra
; MIPS1-BE-NEXT:    nop
entry:
  %0 = load double, ptr @test, align 8
  ret double %0
}

define void @test_swc1(double %a) #0 {
; MIPS1-LE-LABEL: test_swc1:
; MIPS1-LE:       # %bb.0: # %entry
; MIPS1-LE-NEXT:    mfc1 $f0, $f13
; MIPS1-LE-NEXT:    lui	$1, %hi(test)
; MIPS1-LE-NEXT:    swc1 $f0, %lo(test+4)($1)
; MIPS1-LE-NEXT:    mfc1 $f0, $f12
; MIPS1-LE-NEXT:    jr $ra
; MIPS1-LE-NEXT:    swc1 $f0, %lo(test)($1)

; MIPS1-BE-LABEL: test_swc1:
; MIPS1-BE:       # %bb.0: # %entry
; MIPS1-BE-NEXT:    mfc1 $f0, $f12
; MIPS1-BE-NEXT:    lui	$1, %hi(test)
; MIPS1-BE-NEXT:    swc1 $f0, %lo(test+4)($1)
; MIPS1-BE-NEXT:    mfc1 $f0, $f13
; MIPS1-BE-NEXT:    jr $ra
; MIPS1-BE-NEXT:    swc1 $f0, %lo(test)($1)
entry:
  store double %a, ptr @test, align 8
  ret void
}

