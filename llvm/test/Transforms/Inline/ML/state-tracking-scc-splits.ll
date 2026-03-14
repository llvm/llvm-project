; Based on llvm/test/Other/cgscc-iterate-function-mutation.ll
; RUN: opt -passes='default<O3>,print<inline-advisor>' -training-log=/dev/null \
; RUN:   -S -enable-ml-inliner=development -keep-inline-advisor-for-printing < %s 2>&1 | FileCheck %s
; REQUIRES: have_tflite
;
; CHECK: [MLInlineAdvisor] Nodes: 36 Edges: 0

declare void @readnone() nofree nosync readnone
declare void @unknown()
declare void @reference_function_pointer(ptr) nofree nosync readnone

; The @test1_* set of functions checks that when we mutate functions with
; simplifycfg to delete call edges and this ends up splitting both the SCCs
; and the RefSCCs that those functions are in, we re-run the CGSCC passes to
; observe the refined call graph structure.

define void @test1_a() {
  call void @test1_b1()
  call void @test1_b2()
  call void @test1_b3()
  call void @test1_b4()
  ret void
}

define void @test1_b1() {
  call void @readnone()
  ret void
}

define void @test1_b2() {
  call void @readnone()
  br i1 false, label %dead, label %exit

dead:
  call void @test1_a()
  br label %exit

exit:
  ret void
}

define void @test1_b3() {
  call void @unknown()
  br i1 false, label %dead, label %exit

dead:
  call void @test1_a()
  br label %exit

exit:
  ret void
}

define void @test1_b4() {
  call void @readnone()
  br i1 false, label %dead, label %exit

dead:
  call void @test1_a()
  br label %exit

exit:
  ret void
}

define void @test2_a() {
  call void @test2_b1()
  call void @test2_b2()
  call void @test2_b3()
  call void @test2_b4()
  ret void
}

define void @test2_b1() {
  call void @readnone()
  ret void
}

define void @test2_b2() {
  call void @reference_function_pointer(ptr @test2_a)
  br i1 false, label %dead, label %exit

dead:
  call void @test2_a()
  br label %exit

exit:
  ret void
}

define void @test2_b3() {
  call void @reference_function_pointer(ptr @test2_a)
  call void @unknown()
  br i1 false, label %dead, label %exit

dead:
  call void @test2_a()
  br label %exit

exit:
  ret void
}

define void @test2_b4() {
  call void @reference_function_pointer(ptr @test2_a)
  br i1 false, label %dead, label %exit

dead:
  call void @test2_a()
  br label %exit

exit:
  ret void
}

define void @test3_a() {
  call void @test3_b11()
  call void @test3_b21()
  call void @test3_b31()
  call void @test3_b41()
  ret void
}

define void @test3_b11() {
  call void @test3_b12()
  ret void
}

define void @test3_b12() {
  call void @test3_b13()
  ret void
}

define void @test3_b13() {
  call void @readnone()
  ret void
}

define void @test3_b21() {
  call void @test3_b22()
  ret void
}

define void @test3_b22() {
  call void @test3_b23()
  ret void
}

define void @test3_b23() {
  call void @readnone()
  br i1 false, label %dead, label %exit

dead:
  call void @test3_a()
  br label %exit

exit:
  ret void
}

define void @test3_b31() {
  call void @test3_b32()
  ret void
}

define void @test3_b32() {
  call void @test3_b33()
  ret void
}

define void @test3_b33() {
  call void @unknown()
  br i1 false, label %dead, label %exit

dead:
  call void @test3_a()
  br label %exit

exit:
  ret void
}

define void @test3_b41() {
  call void @test3_b42()
  ret void
}

define void @test3_b42() {
  call void @test3_b43()
  ret void
}

define void @test3_b43() {
  call void @readnone()
  br i1 false, label %dead, label %exit

dead:
  call void @test3_a()
  br label %exit

exit:
  ret void
}

define void @test4_a() {
  call void @test4_b11()
  call void @test4_b21()
  call void @test4_b31()
  call void @test4_b41()
  ret void
}

define void @test4_b11() {
  call void @test4_b12()
  ret void
}

define void @test4_b12() {
  call void @test4_b13()
  ret void
}

define void @test4_b13() {
  call void @readnone()
  ret void
}

define void @test4_b21() {
  call void @test4_b22()
  ret void
}

define void @test4_b22() {
  call void @test4_b23()
  ret void
}

define void @test4_b23() {
  call void @reference_function_pointer(ptr @test4_a)
  br i1 false, label %dead, label %exit

dead:
  call void @test4_a()
  br label %exit

exit:
  ret void
}

define void @test4_b31() {
  call void @test4_b32()
  ret void
}

define void @test4_b32() {
  call void @test4_b33()
  ret void
}

define void @test4_b33() {
  call void @reference_function_pointer(ptr @test4_a)
  call void @unknown()
  br i1 false, label %dead, label %exit

dead:
  call void @test4_a()
  br label %exit

exit:
  ret void
}

define void @test4_b41() {
  call void @test4_b42()
  ret void
}

define void @test4_b42() {
  call void @test4_b43()
  ret void
}

define void @test4_b43() {
  call void @reference_function_pointer(ptr @test4_a)
  br i1 false, label %dead, label %exit

dead:
  call void @test4_a()
  br label %exit

exit:
  ret void
}
