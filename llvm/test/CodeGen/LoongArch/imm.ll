; RUN: llc --mtriple=loongarch64 < %s | FileCheck %s

define i64 @imm0() {
; CHECK-LABEL: imm0:
; CHECK:       # %bb.0:
; CHECK-NEXT:    move $a0, $zero
; CHECK-NEXT:    jirl $zero, $ra, 0
  ret i64 0
}

define i64 @imm7ff0000000000000() {
; CHECK-LABEL: imm7ff0000000000000:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lu52i.d $a0, $zero, 2047
; CHECK-NEXT:    jirl $zero, $ra, 0
  ret i64 9218868437227405312
}

define i64 @imm0000000000000fff() {
; CHECK-LABEL: imm0000000000000fff:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ori $a0, $zero, 4095
; CHECK-NEXT:    jirl $zero, $ra, 0
  ret i64 4095
}

define i64 @imm0007ffff00000800() {
; CHECK-LABEL: imm0007ffff00000800:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ori $a0, $zero, 2048
; CHECK-NEXT:    lu32i.d $a0, 524287
; CHECK-NEXT:    jirl $zero, $ra, 0
  ret i64 2251795518720000
}

define i64 @immfff0000000000fff() {
; CHECK-LABEL: immfff0000000000fff:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ori $a0, $zero, 4095
; CHECK-NEXT:    lu52i.d $a0, $a0, -1
; CHECK-NEXT:    jirl $zero, $ra, 0
  ret i64 -4503599627366401
}

define i64 @imm0008000000000fff() {
; CHECK-LABEL: imm0008000000000fff:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ori $a0, $zero, 4095
; CHECK-NEXT:    lu32i.d $a0, -524288
; CHECK-NEXT:    lu52i.d $a0, $a0, 0
; CHECK-NEXT:    jirl $zero, $ra, 0
  ret i64 2251799813689343
}

define i64 @immfffffffffffff800() {
; CHECK-LABEL: immfffffffffffff800:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi.w $a0, $zero, -2048
; CHECK-NEXT:    jirl $zero, $ra, 0
  ret i64 -2048
}

define i64 @imm00000000fffff800() {
; CHECK-LABEL: imm00000000fffff800:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi.w $a0, $zero, -2048
; CHECK-NEXT:    lu32i.d $a0, 0
; CHECK-NEXT:    jirl $zero, $ra, 0
  ret i64 4294965248
}

define i64 @imm000ffffffffff800() {
; CHECK-LABEL: imm000ffffffffff800:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi.w $a0, $zero, -2048
; CHECK-NEXT:    lu52i.d $a0, $a0, 0
; CHECK-NEXT:    jirl $zero, $ra, 0
  ret i64 4503599627368448
}

define i64 @imm00080000fffff800() {
; CHECK-LABEL: imm00080000fffff800:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi.w $a0, $zero, -2048
; CHECK-NEXT:    lu32i.d $a0, -524288
; CHECK-NEXT:    lu52i.d $a0, $a0, 0
; CHECK-NEXT:    jirl $zero, $ra, 0
  ret i64 2251804108650496
}

define i64 @imm000000007ffff000() {
; CHECK-LABEL: imm000000007ffff000:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lu12i.w $a0, 524287
; CHECK-NEXT:    jirl $zero, $ra, 0
  ret i64 2147479552
}

define i64 @imm0000000080000000() {
; CHECK-LABEL: imm0000000080000000:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lu12i.w $a0, -524288
; CHECK-NEXT:    lu32i.d $a0, 0
; CHECK-NEXT:    jirl $zero, $ra, 0
  ret i64 2147483648
}

define i64 @imm000ffffffffff000() {
; CHECK-LABEL: imm000ffffffffff000:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lu12i.w $a0, -1
; CHECK-NEXT:    lu52i.d $a0, $a0, 0
; CHECK-NEXT:    jirl $zero, $ra, 0
  ret i64 4503599627366400
}

define i64 @imm7ff0000080000000() {
; CHECK-LABEL: imm7ff0000080000000:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lu12i.w $a0, -524288
; CHECK-NEXT:    lu32i.d $a0, 0
; CHECK-NEXT:    lu52i.d $a0, $a0, 2047
; CHECK-NEXT:    jirl $zero, $ra, 0
  ret i64 9218868439374888960
}

define i64 @immffffffff80000800() {
; CHECK-LABEL: immffffffff80000800:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lu12i.w $a0, -524288
; CHECK-NEXT:    ori $a0, $a0, 2048
; CHECK-NEXT:    jirl $zero, $ra, 0
  ret i64 -2147481600
}

define i64 @immffffffff7ffff800() {
; CHECK-LABEL: immffffffff7ffff800:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lu12i.w $a0, 524287
; CHECK-NEXT:    ori $a0, $a0, 2048
; CHECK-NEXT:    lu32i.d $a0, -1
; CHECK-NEXT:    jirl $zero, $ra, 0
  ret i64 -2147485696
}

define i64 @imm7fffffff800007ff() {
; CHECK-LABEL: imm7fffffff800007ff:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lu12i.w $a0, -524288
; CHECK-NEXT:    ori $a0, $a0, 2047
; CHECK-NEXT:    lu52i.d $a0, $a0, 2047
; CHECK-NEXT:    jirl $zero, $ra, 0
  ret i64 9223372034707294207
}

define i64 @imm0008000080000800() {
; CHECK-LABEL: imm0008000080000800:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lu12i.w $a0, -524288
; CHECK-NEXT:    ori $a0, $a0, 2048
; CHECK-NEXT:    lu32i.d $a0, -524288
; CHECK-NEXT:    lu52i.d $a0, $a0, 0
; CHECK-NEXT:    jirl $zero, $ra, 0
  ret i64 2251801961170944
}
