; RUN: opt -passes='cgscc(function(instcombine))' -debug-pass-manager -disable-output < %s 2>&1 | FileCheck %s

; We want to run passes on every SCC in a RefSCC without bailing even if one of the SCCs becomes a child SCC.

; This prevents explosive behavior on huge RefSCCs where a non-trivial amount of
; SCCs in the RefSCC become their own RefSCC as passes run on them. Otherwise we
; end up visiting the huge RefSCC the number of times that an SCC is split out
; rather than just twice.

; CHECK: Running pass: InstCombinePass on f1
; CHECK-NOT: InstCombinePass
; CHECK: Running pass: InstCombinePass on f2
; CHECK-NOT: InstCombinePass
; CHECK: Running pass: InstCombinePass on f3
; CHECK-NOT: InstCombinePass
; CHECK: Running pass: InstCombinePass on f4
; CHECK-NOT: InstCombinePass
; CHECK: Running pass: InstCombinePass on f1
; CHECK-NOT: InstCombinePass

@a1 = alias void (), void ()* @f1
@a2 = alias void (), void ()* @f2
@a3 = alias void (), void ()* @f3
@a4 = alias void (), void ()* @f4

define void @f1() {
  call void @a2()
  call void @a3()
  call void @a4()
  ret void
}

define void @f2() {
  call void @a1() readnone nounwind willreturn
  ret void
}

define void @f3() {
  call void @a1() readnone nounwind willreturn
  ret void
}

define void @f4() {
  call void @a1() readnone nounwind willreturn
  ret void
}
