; RUN: llc -march=mipsel < %s | FileCheck %s
; RUN: llc -march=mips64el < %s | FileCheck %s -check-prefixes=CHECK,CHECK64

; CHECK-LABEL: mul5_32:
; CHECK: sll $[[R0:[0-9]+]], $4, 2
; CHECK: addu ${{[0-9]+}}, $[[R0]], $4

define i32 @mul5_32(i32 signext %a) {
entry:
  %mul = mul nsw i32 %a, 5
  ret i32 %mul
}

; CHECK-LABEL:     mul27_32:
; CHECK-DAG: sll $[[R0:[0-9]+]], $4, 2
; CHECK-DAG: addu $[[R1:[0-9]+]], $[[R0]], $4
; CHECK-DAG: sll $[[R2:[0-9]+]], $4, 5
; CHECK:     subu ${{[0-9]+}}, $[[R2]], $[[R1]]

define i32 @mul27_32(i32 signext %a) {
entry:
  %mul = mul nsw i32 %a, 27
  ret i32 %mul
}

; CHECK-LABEL:     muln2147483643_32:
; CHECK-DAG: sll $[[R0:[0-9]+]], $4, 2
; CHECK-DAG: addu $[[R1:[0-9]+]], $[[R0]], $4
; CHECK-DAG: sll $[[R2:[0-9]+]], $4, 31
; CHECK:     addu ${{[0-9]+}}, $[[R2]], $[[R1]]

define i32 @muln2147483643_32(i32 signext %a) {
entry:
  %mul = mul nsw i32 %a, -2147483643
  ret i32 %mul
}

; CHECK64-LABEL:     muln9223372036854775805_64:
; CHECK64-DAG: dsll $[[R0:[0-9]+]], $4, 1
; CHECK64-DAG: daddu $[[R1:[0-9]+]], $[[R0]], $4
; CHECK64-DAG: dsll $[[R2:[0-9]+]], $4, 63
; CHECK64:     daddu ${{[0-9]+}}, $[[R2]], $[[R1]]

define i64 @muln9223372036854775805_64(i64 signext %a) {
entry:
  %mul = mul nsw i64 %a, -9223372036854775805
  ret i64 %mul
}

; CHECK64-LABEL:     muln170141183460469231731687303715884105725_128:
; CHECK64-DAG: dsrl $[[R0:[0-9]+]], $4, 63
; CHECK64-DAG: dsll $[[R1:[0-9]+]], $5, 1
; CHECK64-DAG: or $[[R2:[0-9]+]], $[[R1]], $[[R0]]
; CHECK64-DAG: daddu $[[R3:[0-9]+]], $[[R2]], $5
; CHECK64-DAG: dsll $[[R4:[0-9]+]], $4, 1
; CHECK64-DAG: daddu $[[R5:[0-9]+]], $[[R4]], $4
; CHECK64-DAG: sltu $[[R6:[0-9]+]], $[[R5]], $[[R4]]
; CHECK64-DAG: dsll $[[R7:[0-9]+]], $[[R6]], 32
; CHECK64-DAG: dsrl $[[R8:[0-9]+]], $[[R7]], 32
; CHECK64-DAG: daddu $[[R9:[0-9]+]], $[[R3]], $[[R8]]
; CHECK64-DAG: dsll $[[R10:[0-9]+]], $4, 63
; CHECK64:     daddu ${{[0-9]+}}, $[[R10]], $[[R9]]

define i128 @muln170141183460469231731687303715884105725_128(i128 signext %a) {
entry:
  %mul = mul nsw i128 %a, -170141183460469231731687303715884105725
  ret i128 %mul
}

; CHECK64-LABEL:     mul170141183460469231731687303715884105723_128:
; CHECK64-DAG: dsrl $[[R0:[0-9]+]], $4, 62
; CHECK64-DAG: dsll $[[R1:[0-9]+]], $5, 2
; CHECK64-DAG: or $[[R2:[0-9]+]], $[[R1]], $[[R0]]
; CHECK64-DAG: daddu $[[R3:[0-9]+]], $[[R2]], $5
; CHECK64-DAG: dsll $[[R4:[0-9]+]], $4, 2
; CHECK64-DAG: daddu $[[R5:[0-9]+]], $[[R4]], $4
; CHECK64-DAG: sltu $[[R6:[0-9]+]], $[[R5]], $[[R4]]
; CHECK64-DAG: dsll $[[R7:[0-9]+]], $[[R6]], 32
; CHECK64-DAG: dsrl $[[R8:[0-9]+]], $[[R7]], 32
; CHECK64-DAG: daddu $[[R9:[0-9]+]], $[[R3]], $[[R8]]
; CHECK64-DAG: dsll $[[R10:[0-9]+]], $4, 63
; CHECK64-DAG: dsubu $[[R11:[0-9]+]], $[[R10]], $[[R9]]
; CHECK64-DAG: sltu $[[R12:[0-9]+]], $zero, $[[R5]]
; CHECK64-DAG: dsll $[[R13:[0-9]+]], $[[R12]], 32
; CHECK64-DAG: dsrl $[[R14:[0-9]+]], $[[R13]], 32
; CHECK64-DAG: dsubu $[[R15:[0-9]+]], $[[R11]], $[[R14]]
; CHECK64:     dnegu ${{[0-9]+}}, $[[R5]]

define i128 @mul170141183460469231731687303715884105723_128(i128 signext %a) {
entry:
  %mul = mul nsw i128 %a, 170141183460469231731687303715884105723
  ret i128 %mul
}
