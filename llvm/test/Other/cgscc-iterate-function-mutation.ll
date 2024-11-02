; RUN: opt -aa-pipeline=basic-aa -passes='cgscc(function-attrs,function(simplifycfg))' -S < %s | FileCheck %s

declare void @readnone() nofree nosync readnone
declare void @unknown()
declare void @reference_function_pointer(void()*) nofree nosync readnone

; The @test1_* set of functions checks that when we mutate functions with
; simplifycfg to delete call edges and this ends up splitting both the SCCs
; and the RefSCCs that those functions are in, we re-run the CGSCC passes to
; observe the refined call graph structure.

; CHECK: define void @test1_a() {
define void @test1_a() {
  call void @test1_b1()
  call void @test1_b2()
  call void @test1_b3()
  call void @test1_b4()
  ret void
}

; CHECK: define void @test1_b1() #0 {
define void @test1_b1() {
  call void @readnone()
  ret void
}

; CHECK: define void @test1_b2() #0 {
define void @test1_b2() {
  call void @readnone()
  br i1 false, label %dead, label %exit

dead:
  call void @test1_a()
  br label %exit

exit:
  ret void
}

; CHECK: define void @test1_b3() {
define void @test1_b3() {
  call void @unknown()
  br i1 false, label %dead, label %exit

dead:
  call void @test1_a()
  br label %exit

exit:
  ret void
}

; CHECK: define void @test1_b4() #0 {
define void @test1_b4() {
  call void @readnone()
  br i1 false, label %dead, label %exit

dead:
  call void @test1_a()
  br label %exit

exit:
  ret void
}


; The @test2_* set of functions provide similar checks to @test1_* but only
; splitting the SCCs while leaving the RefSCC intact. This is accomplished by
; having dummy ref edges to the root function.

; CHECK: define void @test2_a() {
define void @test2_a() {
  call void @test2_b1()
  call void @test2_b2()
  call void @test2_b3()
  call void @test2_b4()
  ret void
}

; CHECK: define void @test2_b1() #0 {
define void @test2_b1() {
  call void @readnone()
  ret void
}

; CHECK: define void @test2_b2() #0 {
define void @test2_b2() {
  call void @reference_function_pointer(void()* @test2_a)
  br i1 false, label %dead, label %exit

dead:
  call void @test2_a()
  br label %exit

exit:
  ret void
}

; CHECK: define void @test2_b3() {
define void @test2_b3() {
  call void @reference_function_pointer(void()* @test2_a)
  call void @unknown()
  br i1 false, label %dead, label %exit

dead:
  call void @test2_a()
  br label %exit

exit:
  ret void
}

; CHECK: define void @test2_b4() #0 {
define void @test2_b4() {
  call void @reference_function_pointer(void()* @test2_a)
  br i1 false, label %dead, label %exit

dead:
  call void @test2_a()
  br label %exit

exit:
  ret void
}


; The @test3_* set of functions are the same challenge as @test1_* but with
; multiple layers that have to be traversed in the correct order instead of
; a single node.

; CHECK: define void @test3_a() {
define void @test3_a() {
  call void @test3_b11()
  call void @test3_b21()
  call void @test3_b31()
  call void @test3_b41()
  ret void
}

; CHECK: define void @test3_b11() #0 {
define void @test3_b11() {
  call void @test3_b12()
  ret void
}

; CHECK: define void @test3_b12() #0 {
define void @test3_b12() {
  call void @test3_b13()
  ret void
}

; CHECK: define void @test3_b13() #0 {
define void @test3_b13() {
  call void @readnone()
  ret void
}

; CHECK: define void @test3_b21() #0 {
define void @test3_b21() {
  call void @test3_b22()
  ret void
}

; CHECK: define void @test3_b22() #0 {
define void @test3_b22() {
  call void @test3_b23()
  ret void
}

; CHECK: define void @test3_b23() #0 {
define void @test3_b23() {
  call void @readnone()
  br i1 false, label %dead, label %exit

dead:
  call void @test3_a()
  br label %exit

exit:
  ret void
}

; CHECK: define void @test3_b31() {
define void @test3_b31() {
  call void @test3_b32()
  ret void
}

; CHECK: define void @test3_b32() {
define void @test3_b32() {
  call void @test3_b33()
  ret void
}

; CHECK: define void @test3_b33() {
define void @test3_b33() {
  call void @unknown()
  br i1 false, label %dead, label %exit

dead:
  call void @test3_a()
  br label %exit

exit:
  ret void
}

; CHECK: define void @test3_b41() #0 {
define void @test3_b41() {
  call void @test3_b42()
  ret void
}

; CHECK: define void @test3_b42() #0 {
define void @test3_b42() {
  call void @test3_b43()
  ret void
}

; CHECK: define void @test3_b43() #0 {
define void @test3_b43() {
  call void @readnone()
  br i1 false, label %dead, label %exit

dead:
  call void @test3_a()
  br label %exit

exit:
  ret void
}


; The @test4_* functions exercise the same core challenge as the @test2_*
; functions, but again include long chains instead of single nodes and ensure
; we traverse the chains in the correct order.

; CHECK: define void @test4_a() {
define void @test4_a() {
  call void @test4_b11()
  call void @test4_b21()
  call void @test4_b31()
  call void @test4_b41()
  ret void
}

; CHECK: define void @test4_b11() #0 {
define void @test4_b11() {
  call void @test4_b12()
  ret void
}

; CHECK: define void @test4_b12() #0 {
define void @test4_b12() {
  call void @test4_b13()
  ret void
}

; CHECK: define void @test4_b13() #0 {
define void @test4_b13() {
  call void @readnone()
  ret void
}

; CHECK: define void @test4_b21() #0 {
define void @test4_b21() {
  call void @test4_b22()
  ret void
}

; CHECK: define void @test4_b22() #0 {
define void @test4_b22() {
  call void @test4_b23()
  ret void
}

; CHECK: define void @test4_b23() #0 {
define void @test4_b23() {
  call void @reference_function_pointer(void()* @test4_a)
  br i1 false, label %dead, label %exit

dead:
  call void @test4_a()
  br label %exit

exit:
  ret void
}

; CHECK: define void @test4_b31() {
define void @test4_b31() {
  call void @test4_b32()
  ret void
}

; CHECK: define void @test4_b32() {
define void @test4_b32() {
  call void @test4_b33()
  ret void
}

; CHECK: define void @test4_b33() {
define void @test4_b33() {
  call void @reference_function_pointer(void()* @test4_a)
  call void @unknown()
  br i1 false, label %dead, label %exit

dead:
  call void @test4_a()
  br label %exit

exit:
  ret void
}

; CHECK: define void @test4_b41() #0 {
define void @test4_b41() {
  call void @test4_b42()
  ret void
}

; CHECK: define void @test4_b42() #0 {
define void @test4_b42() {
  call void @test4_b43()
  ret void
}

; CHECK: define void @test4_b43() #0 {
define void @test4_b43() {
  call void @reference_function_pointer(void()* @test4_a)
  br i1 false, label %dead, label %exit

dead:
  call void @test4_a()
  br label %exit

exit:
  ret void
}

; CHECK: attributes #0 = { nofree nosync memory(none) }
