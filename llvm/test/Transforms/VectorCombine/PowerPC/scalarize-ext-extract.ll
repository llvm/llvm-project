; RUN: opt -passes='vector-combine,dce' -S -mtriple=aarch64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=LE
; RUN: opt -passes='vector-combine,dce' -S -mtriple=powerpc64-ibm-aix-xcoff     %s -o - | FileCheck %s --check-prefix=BE

define i64 @g(<8 x i8> %v) {
  %z  = zext <8 x i8> %v to <8 x i64>
  %e0 = extractelement <8 x i64> %z, i32 0
  %e7 = extractelement <8 x i64> %z, i32 7
  %sum = add i64 %e0, %e7
  ret i64 %sum
}

; LE-LABEL: @g(
; LE: bitcast <8 x i8> %{{.*}} to i64
; LE: lshr i64 %{{.*}}, 56
; LE: and i64 %{{.*}}, 255
; LE-NOT: extractelement

; BE-LABEL: @g(
; BE: bitcast <8 x i8> %{{.*}} to i64
; BE: and i64 %{{.*}}, 255
; BE: lshr i64 %{{.*}}, 56
; BE-NOT: extractelement

