; RUN: opt < %s -passes='print<cost-model>' 2>&1 -disable-output -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:       -mcpu=future | FileCheck %s --check-prefix=FUTURE
; RUN: opt < %s -passes='print<cost-model>' 2>&1 -disable-output -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:       -mcpu=pwr9 | FileCheck %s --check-prefix=PWR9

define void @test(i16 %p1, i16 %p2, <4 x i16> %p3, <4 x i16> %p4) {
  %i1 = add i16 %p1, %p2
  %v1 = add <4 x i16> %p3, %p4
  ret void
  ; FUTURE: cost of 1 {{.*}} add
  ; FUTURE: cost of 1 {{.*}} add

  ; PWR9: cost of 1 {{.*}} add
  ; PWR9: cost of 2 {{.*}} add
}

