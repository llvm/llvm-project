; Test the behavior of GlobalDCE in conjunction with comdats.
;
; RUN: opt < %s -globaldce -S | FileCheck %s

; First test checks that if one function in a comdat group is used, both other
; functions and other globals even if unused will be preserved.
$test1_c = comdat any
; CHECK: $test1_c = comdat any

; Second test checks that if one function in a comdat group is used, both other
; functions and other globals even if unused will be preserved.
$test2_c = comdat any
; CHECK: $test2_c = comdat any

; Third test checks that calling a function in a comdat group with an alias
; preserves the alias.
$test3_c = comdat any
; CHECK: $test3_c = comdat any

; Fourth test checks that calling an alias in a comdat group with a function
; preserves the function. (This is the trivial case as the alias uses the
; function.)
$test4_c = comdat any
; CHECK: $test4_c = comdat any

; Fifth test checks that calling a function in a comdat group that is used as
; the resolver of an ifunc doesn't preserve that ifunc. ifunc symbols don't
; participate in the comdat group of their resolver function as they are
; considered separate objects.
$test5_c = comdat any
; CHECK: $test5_c = comdat any

; Sixth test checks that calling an ifunc whose resolver is in a comdat group
; preserves the resolver. This is the trivial case as the ifunc uses the
; resolver.
$test6_c = comdat any
; CHECK: $test6_c = comdat any

; Seventh test checks that we can eliminate a comdat when it has only one dead function participant.
$test7_c = comdat any
; CHECK-NOT: $test7_c = comdat any

; Eighth test checks that we can eliminate a comdat when it has only one dead global participant.
$test8_c = comdat any
; CHECK-NOT: $test8_c = comdat any

; Ninth test checks that we can eliminate a comdat when there are multiple
; dead participants.
$test9_c = comdat any
; CHECK-NOT: $test9_c = comdat any

; Tenth test checks that we can eliminate a comdat when it has multiple
; participants that form internal cyclic uses but are never used externally and
; thus the entire ifunc can safely be eliminated.
$test10_c = comdat any
; CHECK-NOT: $test10_c = comdat any

@test1_gv = linkonce_odr unnamed_addr global i32 42, comdat($test1_c)
; CHECK: @test1_gv = linkonce_odr unnamed_addr global

@test2_used = linkonce_odr unnamed_addr global i32 42, comdat($test2_c)
; CHECK: @test2_used = linkonce_odr unnamed_addr global

@test2_gv = linkonce_odr unnamed_addr global i32 42, comdat($test2_c)
; CHECK: @test2_gv = linkonce_odr unnamed_addr global

@test8_gv = linkonce_odr unnamed_addr global i32 42, comdat($test8_c)
; CHECK-NOT: @test8_gv

@test9_gv = linkonce_odr unnamed_addr global i32 42, comdat($test9_c)
; CHECK-NOT: @test9_gv

@test10_gv = linkonce_odr unnamed_addr global void ()* @test10_f, comdat($test10_c)
; CHECK-NOT: @test10_gv

@test3_a = linkonce_odr unnamed_addr alias void (), void ()* @test3_f
; CHECK: @test3_a = linkonce_odr unnamed_addr alias

@test4_a = linkonce_odr unnamed_addr alias void (), void ()* @test4_f
; CHECK: @test4_a = linkonce_odr unnamed_addr alias

@test10_a = linkonce_odr unnamed_addr alias void (), void ()* @test10_g
; CHECK-NOT: @test10_a

@test5_if = linkonce_odr ifunc void (), void ()* ()* @test5_f
; CHECK-NOT: @test5_if

@test6_if = linkonce_odr ifunc void (), void ()* ()* @test6_f
; CHECK: @test6_if = linkonce_odr ifunc

; This function is directly used and so cannot be eliminated.
define linkonce_odr void @test1_used() comdat($test1_c) {
; CHECK: define linkonce_odr void @test1_used()
entry:
  ret void
}

define linkonce_odr void @test1_f() comdat($test1_c) {
; CHECK: define linkonce_odr void @test1_f()
entry:
  ret void
}

; Now test that a function, global variable, alias, and ifunc in the same
; comdat are kept.
define linkonce_odr void @test2_f() comdat($test2_c) {
; CHECK: define linkonce_odr void @test2_f()
entry:
  ret void
}

define linkonce_odr void @test3_f() comdat($test3_c) {
; CHECK: define linkonce_odr void @test3_f()
entry:
  ret void
}

define linkonce_odr void @test4_f() comdat($test4_c) {
; CHECK: define linkonce_odr void @test4_f()
entry:
  ret void
}

declare void @test_external()

define linkonce_odr void ()* @test5_f() comdat($test5_c) {
; CHECK: define linkonce_odr void ()* @test5_f()
entry:
  ret void ()* @test_external
}

define linkonce_odr void ()* @test6_f() comdat($test6_c) {
; CHECK: define linkonce_odr void ()* @test6_f()
entry:
  ret void ()* @test_external
}

define linkonce_odr void @test7_f() comdat($test7_c) {
; CHECK-NOT: @test7_f
entry:
  ret void
}

define linkonce_odr void @test9_f() comdat($test9_c) {
; CHECK-NOT: @test9_f
entry:
  ret void
}

define linkonce_odr void @test10_f() comdat($test10_c) {
; CHECK-NOT: @test10_f
entry:
  %gv = load void ()*, void ()** @test10_gv
  call void @test10_a()
  ret void
}

define linkonce_odr void @test10_g() comdat($test10_c) {
; CHECK-NOT: @test10_g
entry:
  call void @test10_f()
  ret void
}


; An external function to pin as "used" various things above that shouldn't be
; eliminated.
define void @external_user() {
  call void @test1_used()
  %gv = load i32, i32* @test2_used

  call void @test3_f()
  call void @test4_a()

  %fptr = call void() *@test5_f()
  call void @test6_if()
  ret void
}
