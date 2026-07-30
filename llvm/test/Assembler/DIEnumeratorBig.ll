;; Round-trip test for enumeration members using more than 64 bits.

; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

!named = !{!0, !1, !2}

; CHECK: !DIEnumerator(name: "D0", value: -170141183460469231731687303715884105728)
; CHECK: !DIEnumerator(name: "D1", value: 170141183460469231731687303715884105727)
!0 = !DIEnumerator(name: "D0", value: -170141183460469231731687303715884105728)
!1 = !DIEnumerator(name: "D1", value: 170141183460469231731687303715884105727)

; CHECK: !DIEnumerator(name: "D1", value: 2722258935367507707706996859454145691648, isUnsigned: true)
!2 = !DIEnumerator(name: "D1", value: 2722258935367507707706996859454145691648, isUnsigned: true)
