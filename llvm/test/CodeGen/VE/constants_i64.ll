; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define i64 @p0i64() {
; CHECK-LABEL: p0i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:  or %s0, 0, (0)1
; CHECK-NEXT:  or %s11, 0, %s9
  ret i64 0
}

define signext i64 @p0si64() {
; CHECK-LABEL: p0si64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:  or %s0, 0, (0)1
; CHECK-NEXT:  or %s11, 0, %s9
  ret i64 0
}

define zeroext i64 @p0zi64() {
; CHECK-LABEL: p0zi64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:  or %s0, 0, (0)1
; CHECK-NEXT:  or %s11, 0, %s9
  ret i64 0
}

define i64 @p128i64() {
; CHECK-LABEL: p128i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:  lea %s0, 128
; CHECK-NEXT:  or %s11, 0, %s9
  ret i64 128
}

define signext i64 @p128si64() {
; CHECK-LABEL: p128si64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:  lea %s0, 128
; CHECK-NEXT:  or %s11, 0, %s9
  ret i64 128
}

define zeroext i64 @p128zi64() {
; CHECK-LABEL: p128zi64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:  lea %s0, 128
; CHECK-NEXT:  or %s11, 0, %s9
  ret i64 128
}

define i64 @p2264924160i64() {
; CHECK-LABEL: p2264924160i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:  lea %s0, -2030043136
; CHECK-NEXT:  and %s0, %s0, (32)0
; CHECK-NEXT:  or %s11, 0, %s9
  ret i64 2264924160
}

define signext i64 @p2264924160si64() {
; CHECK-LABEL: p2264924160si64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:  lea %s0, -2030043136
; CHECK-NEXT:  and %s0, %s0, (32)0
; CHECK-NEXT:  or %s11, 0, %s9
  ret i64 2264924160
}

define zeroext i64 @p2264924160zi64() {
; CHECK-LABEL: p2264924160zi64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:  lea %s0, -2030043136
; CHECK-NEXT:  and %s0, %s0, (32)0
; CHECK-NEXT:  or %s11, 0, %s9
  ret i64 2264924160
}

define i64 @p2147483647i64() {
; CHECK-LABEL: p2147483647i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:  lea %s0, 2147483647
; CHECK-NEXT:  or %s11, 0, %s9
  ret i64 2147483647
}

define signext i64 @p2147483647si64() {
; CHECK-LABEL: p2147483647si64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:  lea %s0, 2147483647
; CHECK-NEXT:  or %s11, 0, %s9
  ret i64 2147483647
}

define zeroext i64 @p2147483647zi64() {
; CHECK-LABEL: p2147483647zi64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:  lea %s0, 2147483647
; CHECK-NEXT:  or %s11, 0, %s9
  ret i64 2147483647
}

define i64 @p15032385535i64() {
; CHECK-LABEL: p15032385535i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:  lea %s0, 2147483647
; CHECK-NEXT:  lea.sl %s0, 3(%s0)
; CHECK-NEXT:  or %s11, 0, %s9
  ret i64 15032385535
}

define signext i64 @p15032385535si64() {
; CHECK-LABEL: p15032385535si64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:  lea %s0, 2147483647
; CHECK-NEXT:  lea.sl %s0, 3(%s0)
; CHECK-NEXT:  or %s11, 0, %s9
  ret i64 15032385535
}

define zeroext i64 @p15032385535zi64() {
; CHECK-LABEL: p15032385535zi64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:  lea %s0, 2147483647
; CHECK-NEXT:  lea.sl %s0, 3(%s0)
; CHECK-NEXT:  or %s11, 0, %s9
  ret i64 15032385535
}

define i64 @p15032385536i64() {
; CHECK-LABEL: p15032385536i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:  lea %s0, -2147483648
; CHECK-NEXT:  and %s0, %s0, (32)0
; CHECK-NEXT:  lea.sl %s0, 3(%s0)
; CHECK-NEXT:  or %s11, 0, %s9
  ret i64 15032385536
}

define signext i64 @p15032385536si64() {
; CHECK-LABEL: p15032385536si64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:  lea %s0, -2147483648
; CHECK-NEXT:  and %s0, %s0, (32)0
; CHECK-NEXT:  lea.sl %s0, 3(%s0)
; CHECK-NEXT:  or %s11, 0, %s9
  ret i64 15032385536
}

define zeroext i64 @p15032385536zi64() {
; CHECK-LABEL: p15032385536zi64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:  lea %s0, -2147483648
; CHECK-NEXT:  and %s0, %s0, (32)0
; CHECK-NEXT:  lea.sl %s0, 3(%s0)
; CHECK-NEXT:  or %s11, 0, %s9
  ret i64 15032385536
}
