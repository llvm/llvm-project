; RUN: llc -march=mipsel -relocation-model=pic < %s | FileCheck %s


; Check that function accesses vector return value from stack in cases when
; vector can't be returned in registers. Also check that caller passes in
; register $4 stack address where the vector should be placed.


declare <8 x i32>    @i8(...)
declare <4 x float>  @f4(...)
declare <4 x double> @d4(...)

define i32 @call_i8() {
entry:
  %call = call <8 x i32> (...) @i8()
  %v0 = extractelement <8 x i32> %call, i32 0
  %v1 = extractelement <8 x i32> %call, i32 1
  %v2 = extractelement <8 x i32> %call, i32 2
  %v3 = extractelement <8 x i32> %call, i32 3
  %v4 = extractelement <8 x i32> %call, i32 4
  %v5 = extractelement <8 x i32> %call, i32 5
  %v6 = extractelement <8 x i32> %call, i32 6
  %v7 = extractelement <8 x i32> %call, i32 7
  %add1 = add i32 %v0, %v1
  %add2 = add i32 %v2, %v3
  %add3 = add i32 %v4, %v5
  %add4 = add i32 %v6, %v7
  %add5 = add i32 %add1, %add2
  %add6 = add i32 %add3, %add4
  %add7 = add i32 %add5, %add6
  ret i32 %add7

; CHECK-LABEL:        call_i8:
; CHECK:        call16(i8)
; CHECK:        addiu   $4, $sp, 32
; CHECK:        lw      $[[R0:[a-z0-9]+]], 60($sp)
; CHECK:        lw      $[[R1:[a-z0-9]+]], 56($sp)
; CHECK:        lw      $[[R2:[a-z0-9]+]], 52($sp)
; CHECK:        lw      $[[R3:[a-z0-9]+]], 48($sp)
; CHECK:        lw      $[[R4:[a-z0-9]+]], 44($sp)
; CHECK:        lw      $[[R5:[a-z0-9]+]], 40($sp)
; CHECK:        lw      $[[R6:[a-z0-9]+]], 36($sp)
; CHECK:        lw      $[[R7:[a-z0-9]+]], 32($sp)
}


define float @call_f4() {
entry:
  %call = call <4 x float> (...) @f4()
  %v0 = extractelement <4 x float> %call, i32 0
  %v1 = extractelement <4 x float> %call, i32 1
  %v2 = extractelement <4 x float> %call, i32 2
  %v3 = extractelement <4 x float> %call, i32 3
  %add1 = fadd float %v0, %v1
  %add2 = fadd float %v2, %v3
  %add3 = fadd float %add1, %add2
  ret float %add3

; CHECK-LABEL:        call_f4:
; CHECK:        call16(f4)
; CHECK:        addiu   $4, $sp, 16
; CHECK:        lwc1    $[[R0:[a-z0-9]+]], 28($sp)
; CHECK:        lwc1    $[[R1:[a-z0-9]+]], 24($sp)
; CHECK:        lwc1    $[[R3:[a-z0-9]+]], 20($sp)
; CHECK:        lwc1    $[[R4:[a-z0-9]+]], 16($sp)
}


define double @call_d4() {
entry:
  %call = call <4 x double> (...) @d4()
  %v0 = extractelement <4 x double> %call, i32 0
  %v1 = extractelement <4 x double> %call, i32 1
  %v2 = extractelement <4 x double> %call, i32 2
  %v3 = extractelement <4 x double> %call, i32 3
  %add1 = fadd double %v0, %v1
  %add2 = fadd double %v2, %v3
  %add3 = fadd double %add1, %add2
  ret double %add3

; CHECK-LABEL:        call_d4:
; CHECK:        call16(d4)
; CHECK:        addiu   $4, $sp, 32
; CHECK:        ldc1    $[[R0:[a-z0-9]+]], 56($sp)
; CHECK:        ldc1    $[[R1:[a-z0-9]+]], 48($sp)
; CHECK:        ldc1    $[[R3:[a-z0-9]+]], 40($sp)
; CHECK:        ldc1    $[[R4:[a-z0-9]+]], 32($sp)
}



; Check that function accesses vector return value from registers in cases when
; vector can be returned in registers


declare <4 x i32>    @i4(...)
declare <2 x float>  @f2(...)
declare <2 x double> @d2(...)

define i32 @call_i4() {
entry:
  %call = call <4 x i32> (...) @i4()
  %v0 = extractelement <4 x i32> %call, i32 0
  %v1 = extractelement <4 x i32> %call, i32 1
  %v2 = extractelement <4 x i32> %call, i32 2
  %v3 = extractelement <4 x i32> %call, i32 3
  %add1 = add i32 %v0, %v1
  %add2 = add i32 %v2, %v3
  %add3 = add i32 %add1, %add2
  ret i32 %add3

; CHECK-LABEL:        call_i4:
; CHECK:        call16(i4)
; CHECK-NOT:    lw
; CHECK:        addu    $[[R2:[a-z0-9]+]], $[[R0:[a-z0-9]+]], $[[R1:[a-z0-9]+]]
; CHECK:        addu    $[[R5:[a-z0-9]+]], $[[R3:[a-z0-9]+]], $[[R4:[a-z0-9]+]]
; CHECK:        addu    $[[R6:[a-z0-9]+]], $[[R5]], $[[R2]]
}


define float @call_f2() {
entry:
  %call = call <2 x float> (...) @f2()
  %v0 = extractelement <2 x float> %call, i32 0
  %v1 = extractelement <2 x float> %call, i32 1
  %add1 = fadd float %v0, %v1
  ret float %add1

; CHECK-LABEL:        call_f2:
; CHECK:        call16(f2)
; CHECK-NOT:    lwc1
; CHECK:        add.s    $[[R2:[a-z0-9]+]], $[[R0:[a-z0-9]+]], $[[R1:[a-z0-9]+]]
}


define double @call_d2() {
entry:
  %call = call <2 x double> (...) @d2()
  %v0 = extractelement <2 x double> %call, i32 0
  %v1 = extractelement <2 x double> %call, i32 1
  %add1 = fadd double %v0, %v1
  ret double %add1

; CHECK-LABEL:        call_d2:
; CHECK:        call16(d2)
; CHECK-NOT:    ldc1
; CHECK:        add.d    $[[R2:[a-z0-9]+]], $[[R0:[a-z0-9]+]], $[[R1:[a-z0-9]+]]
}



; Check that function returns vector on stack in cases when vector can't be
; returned in registers. Also check that vector is placed on stack starting
; from the address in register $4.


define <8 x i32> @return_i8() {
entry:
  ret <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>

; CHECK-LABEL:        return_i8:
; CHECK:        sw      $[[R0:[a-z0-9]+]], 28($4)
; CHECK:        sw      $[[R1:[a-z0-9]+]], 24($4)
; CHECK:        sw      $[[R2:[a-z0-9]+]], 20($4)
; CHECK:        sw      $[[R3:[a-z0-9]+]], 16($4)
; CHECK:        sw      $[[R4:[a-z0-9]+]], 12($4)
; CHECK:        sw      $[[R5:[a-z0-9]+]], 8($4)
; CHECK:        sw      $[[R6:[a-z0-9]+]], 4($4)
; CHECK:        sw      $[[R7:[a-z0-9]+]], 0($4)
}


define <4 x float> @return_f4(float %a, float %b, float %c, float %d) {
entry:
  %vecins1 = insertelement <4 x float> undef,    float %a, i32 0
  %vecins2 = insertelement <4 x float> %vecins1, float %b, i32 1
  %vecins3 = insertelement <4 x float> %vecins2, float %c, i32 2
  %vecins4 = insertelement <4 x float> %vecins3, float %d, i32 3
  ret <4 x float> %vecins4

; CHECK-LABEL:        return_f4:
; CHECK-DAG:    lwc1    $[[R0:[a-z0-9]+]], 16($sp)
; CHECK-DAG:    swc1    $[[R0]], 12($4)
; CHECK-DAG:    sw      $7, 8($4)
; CHECK-DAG:    sw      $6, 4($4)
; CHECK-DAG:    sw      $5, 0($4)
}


define <4 x double> @return_d4(double %a, double %b, double %c, double %d) {
entry:
  %vecins1 = insertelement <4 x double> undef,    double %a, i32 0
  %vecins2 = insertelement <4 x double> %vecins1, double %b, i32 1
  %vecins3 = insertelement <4 x double> %vecins2, double %c, i32 2
  %vecins4 = insertelement <4 x double> %vecins3, double %d, i32 3
  ret <4 x double> %vecins4

; CHECK-LABEL:            return_d4:
; CHECK-DAG:        sdc1    $[[R0:[a-z0-9]+]], 24($4)
; CHECK-DAG:        sdc1    $[[R1:[a-z0-9]+]], 16($4)
; CHECK-DAG:        sdc1    $[[R2:[a-z0-9]+]], 8($4)
; CHECK-DAG:        sdc1    $[[R3:[a-z0-9]+]], 0($4)
}



; Check that function returns vector in registers in cases when vector can be
; returned in registers.


define <4 x i32> @return_i4() {
entry:
  ret <4 x i32> <i32 0, i32 1, i32 2, i32 3>

; CHECK-LABEL:        return_i4:
; CHECK:        addiu   $2, $zero, 0
; CHECK:        addiu   $3, $zero, 1
; CHECK:        addiu   $4, $zero, 2
; CHECK:        addiu   $5, $zero, 3
}


define <2 x float> @return_f2(float %a, float %b) {
entry:
  %vecins1 = insertelement <2 x float> undef,    float %a, i32 0
  %vecins2 = insertelement <2 x float> %vecins1, float %b, i32 1
  ret <2 x float> %vecins2

; CHECK-LABEL:        return_f2:
; CHECK:        mov.s   $f0, $f12
; CHECK:        mov.s   $f2, $f14
}


define <2 x double> @return_d2(double %a, double %b) {
entry:
  %vecins1 = insertelement <2 x double> undef,    double %a, i32 0
  %vecins2 = insertelement <2 x double> %vecins1, double %b, i32 1
  ret <2 x double> %vecins2

; CHECK-LABEL:        return_d2:
; CHECK:        mov.d   $f0, $f12
; CHECK:        mov.d   $f2, $f14
}
