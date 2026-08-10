; RUN: llc -verify-machineinstrs < %s | FileCheck --check-prefix=P8 --check-prefix=CHECK %s
; RUN: llc -mcpu=pwr9 -verify-machineinstrs < %s | FileCheck --check-prefix=P9 --check-prefix=CHECK %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; CHECK-LABEL: lshr:
; CHECK-DAG: subfic [[R0:[0-9]+]], 5, 64
; CHECK-DAG: addi [[R1:[0-9]+]], 5, -64
; CHECK-DAG: srd [[R2:[0-9]+]], 3, 5
; CHECK-DAG: sld [[R3:[0-9]+]], 4, [[R0]]
; CHECK-DAG: srd [[R4:[0-9]+]], 4, [[R1]]
; CHECK-DAG: or [[R5:[0-9]+]], [[R2]], [[R3]]
; CHECK-DAG: or 3, [[R5]], [[R4]]
; CHECK-DAG: srd 4, 4, 5
; CHECK: blr
define i128 @lshr(i128 %x, i128 %y) {
  %r = lshr i128 %x, %y
  ret i128 %r
}
; CHECK-LABEL: ashr:
; CHECK-DAG: subfic [[R0:[0-9]+]], 5, 64
; CHECK-DAG: addi [[R1:[0-9]+]], 5, -64
; CHECK-DAG: srd [[R2:[0-9]+]], 3, 5
; CHECK-DAG: sld [[R3:[0-9]+]], 4, [[R0]]
; CHECK-DAG: srad [[R4:[0-9]+]], 4, [[R1]]
; CHECK-DAG: or [[R5:[0-9]+]], [[R2]], [[R3]]
; CHECK-DAG: cmpwi [[R1]], 1
; CHECK-DAG: srad 4, 4, 5
; CHECK-DAG: isellt 3, [[R5]], [[R4]]
; CHECK: blr
define i128 @ashr(i128 %x, i128 %y) {
  %r = ashr i128 %x, %y
  ret i128 %r
}
; CHECK-LABEL: shl:
; CHECK-DAG: subfic [[R0:[0-9]+]], 5, 64
; CHECK-DAG: addi [[R1:[0-9]+]], 5, -64
; CHECK-DAG: sld [[R2:[0-9]+]], 4, 5
; CHECK-DAG: srd [[R3:[0-9]+]], 3, [[R0]]
; CHECK-DAG: sld [[R4:[0-9]+]], 3, [[R1]]
; CHECK-DAG: or [[R5:[0-9]+]], [[R2]], [[R3]]
; CHECK-DAG: or 4, [[R5]], [[R4]]
; CHECK-DAG: sld 3, 3, 5
; CHECK: blr
define i128 @shl(i128 %x, i128 %y) {
  %r = shl i128 %x, %y
  ret i128 %r
}

; CHECK-LABEL: shl_v1i128:
; P8-NOT: {{\b}}vslo
; P8-NOT: {{\b}}vsl
; P9-DAG: vslo
; P9-DAG: vspltb
; P9: vsl
; P9-NOT: {{\b}}sld
; P9-NOT: {{\b}}srd
; CHECK: blr
define i128 @shl_v1i128(i128 %arg, i128 %amt) local_unnamed_addr #0 {
entry:
  %0 = insertelement <1 x i128> undef, i128 %arg, i32 0
  %1 = insertelement <1 x i128> undef, i128 %amt, i32 0
  %2 = shl <1 x i128> %0, %1
  %retval = extractelement <1 x i128> %2, i32 0
  ret i128 %retval
}

; CHECK-LABEL: lshr_v1i128:
; P8-NOT: {{\b}}vsro
; P8-NOT: {{\b}}vsr
; P9-DAG: vsro
; P9-DAG: vspltb
; P9: vsr
; P9-NOT: {{\b}}srd
; P9-NOT: {{\b}}sld
; CHECK: blr
define i128 @lshr_v1i128(i128 %arg, i128 %amt) local_unnamed_addr #0 {
entry:
  %0 = insertelement <1 x i128> undef, i128 %arg, i32 0
  %1 = insertelement <1 x i128> undef, i128 %amt, i32 0
  %2 = lshr <1 x i128> %0, %1
  %retval = extractelement <1 x i128> %2, i32 0
  ret i128 %retval
}

; Arithmetic shift right is not available as an operation on the vector registers.
; CHECK-LABEL: ashr_v1i128:
; CHECK-NOT: {{\b}}vsro
; CHECK-NOT: {{\b}}vsr
; CHECK: blr
define i128 @ashr_v1i128(i128 %arg, i128 %amt) local_unnamed_addr #0 {
entry:
  %0 = insertelement <1 x i128> undef, i128 %arg, i32 0
  %1 = insertelement <1 x i128> undef, i128 %amt, i32 0
  %2 = ashr <1 x i128> %0, %1
  %retval = extractelement <1 x i128> %2, i32 0
  ret i128 %retval
}
