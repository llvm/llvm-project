; RUN: opt -attributor -attributor-manifest-internal -attributor-disable=false -attributor-max-iterations-verify -attributor-max-iterations=6 -S < %s | FileCheck %s --check-prefix=ATTRIBUTOR

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Test cases specifically designed for "align" attribute.
; We use FIXME's to indicate problems and missing attributes.


; TEST 1
; ATTRIBUTOR: define align 8 i32* @test1(i32* readnone returned align 8 "no-capture-maybe-returned" %0)
define i32* @test1(i32* align 8 %0) #0 {
  ret i32* %0
}

; TEST 2
; ATTRIBUTOR: define i32* @test2(i32* readnone returned "no-capture-maybe-returned" %0)
define i32* @test2(i32* %0) #0 {
  ret i32* %0
}

; TEST 3
; ATTRIBUTOR: define align 4 i32* @test3(i32* readnone align 8 "no-capture-maybe-returned" %0, i32* readnone align 4 "no-capture-maybe-returned" %1, i1 %2)
define i32* @test3(i32* align 8 %0, i32* align 4 %1, i1 %2) #0 {
  %ret = select i1 %2, i32* %0, i32* %1
  ret i32* %ret
}

; TEST 4
; ATTRIBUTOR: define align 32 i32* @test4(i32* readnone align 32 "no-capture-maybe-returned" %0, i32* readnone align 32 "no-capture-maybe-returned" %1, i1 %2)
define i32* @test4(i32* align 32 %0, i32* align 32 %1, i1 %2) #0 {
  %ret = select i1 %2, i32* %0, i32* %1
  ret i32* %ret
}

; TEST 5
declare i32* @unknown()
declare align 8 i32* @align8()


; ATTRIBUTOR: define align 8 i32* @test5_1()
define i32* @test5_1() {
  %ret = tail call align 8 i32* @unknown()
  ret i32* %ret
}

; ATTRIBUTOR: define align 8 i32* @test5_2()
define i32* @test5_2() {
  %ret = tail call i32* @align8()
  ret i32* %ret
}

; TEST 6
; SCC
; ATTRIBUTOR: define noalias nonnull align 536870912 dereferenceable(4294967295) i32* @test6_1()
define i32* @test6_1() #0 {
  %ret = tail call i32* @test6_2()
  ret i32* %ret
}

; ATTRIBUTOR: define noalias nonnull align 536870912 dereferenceable(4294967295) i32* @test6_2()
define i32* @test6_2() #0 {
  %ret = tail call i32* @test6_1()
  ret i32* %ret
}


; char a1 __attribute__((aligned(8)));
; char a2 __attribute__((aligned(16)));
;
; char* f1(char* a ){
;     return a?a:f2(&a1);
; }
; char* f2(char* a){
;     return a?f1(a):f3(&a2);
; }
;
; char* f3(char* a){
;     return a?&a1: f1(&a2);
; }

@a1 = common global i8 0, align 8
@a2 = common global i8 0, align 16

; Function Attrs: nounwind readnone ssp uwtable
define internal i8* @f1(i8* readnone %0) local_unnamed_addr #0 {
; ATTRIBUTOR: define internal nonnull align 8 dereferenceable(1) i8* @f1(i8* nonnull readnone align 8 dereferenceable(1) "no-capture-maybe-returned" %0)
  %2 = icmp eq i8* %0, null
  br i1 %2, label %3, label %5

; <label>:3:                                      ; preds = %1
; ATTRIBUTOR: %4 = tail call align 8 i8* @f2(i8* nonnull align 8 dereferenceable(1) @a1)
  %4 = tail call i8* @f2(i8* nonnull @a1)
; ATTRIBUTOR: %l = load i8, i8* %4, align 8
  %l = load i8, i8* %4
  br label %5

; <label>:5:                                      ; preds = %1, %3
  %6 = phi i8* [ %4, %3 ], [ %0, %1 ]
  ret i8* %6
}

; Function Attrs: nounwind readnone ssp uwtable
define internal i8* @f2(i8* readnone %0) local_unnamed_addr #0 {
; ATTRIBUTOR: define internal nonnull align 8 dereferenceable(1) i8* @f2(i8* nonnull readnone align 8 dereferenceable(1) "no-capture-maybe-returned" %0)
  %2 = icmp eq i8* %0, null
  br i1 %2, label %5, label %3

; <label>:3:                                      ; preds = %1

; ATTRIBUTOR: %4 = tail call i8* @f1(i8* nonnull align 8 dereferenceable(1) "no-capture-maybe-returned" @a1)
  %4 = tail call i8* @f1(i8* nonnull %0)
  br label %7

; <label>:5:                                      ; preds = %1
; ATTRIBUTOR: %6 = tail call i8* @f3(i8* nonnull align 16 dereferenceable(1) @a2)
  %6 = tail call i8* @f3(i8* nonnull @a2)
  br label %7

; <label>:7:                                      ; preds = %5, %3
  %8 = phi i8* [ %4, %3 ], [ %6, %5 ]
  ret i8* %8
}

; Function Attrs: nounwind readnone ssp uwtable
define internal i8* @f3(i8* readnone %0) local_unnamed_addr #0 {
; ATTRIBUTOR: define internal nonnull align 8 dereferenceable(1) i8* @f3(i8* nocapture nonnull readnone align 16 dereferenceable(1) %0)
  %2 = icmp eq i8* %0, null
  br i1 %2, label %3, label %5

; <label>:3:                                      ; preds = %1
; ATTRIBUTOR: %4 = tail call i8* @f1(i8* nonnull align 16 dereferenceable(1) @a2)
  %4 = tail call i8* @f1(i8* nonnull @a2)
  br label %5

; <label>:5:                                      ; preds = %1, %3
  %6 = phi i8* [ %4, %3 ], [ @a1, %1 ]
  ret i8* %6
}

; TEST 7
; Better than IR information
; ATTRIBUTOR: define align 32 i32* @test7(i32* readnone returned align 32 "no-capture-maybe-returned" %p)
define align 4 i32* @test7(i32* align 32 %p) #0 {
  tail call i8* @f1(i8* align 8 dereferenceable(1) @a1)
  ret i32* %p
}


; TEST 8
define void @test8_helper() {
  %ptr0 = tail call i32* @unknown()
  %ptr1 = tail call align 4 i32* @unknown()
  %ptr2 = tail call align 8 i32* @unknown()

  tail call void @test8(i32* %ptr1, i32* %ptr1, i32* %ptr0)
; ATTRIBUTOR: tail call void @test8(i32* align 4 %ptr1, i32* align 4 %ptr1, i32* %ptr0)
  tail call void @test8(i32* %ptr2, i32* %ptr1, i32* %ptr1)
; ATTRIBUTOR: tail call void @test8(i32* align 8 %ptr2, i32* align 4 %ptr1, i32* align 4 %ptr1)
  tail call void @test8(i32* %ptr2, i32* %ptr1, i32* %ptr1)
; ATTRIBUTOR: tail call void @test8(i32* align 8 %ptr2, i32* align 4 %ptr1, i32* align 4 %ptr1)
  ret void
}

define internal void @test8(i32* %a, i32* %b, i32* %c) {
; ATTRIBUTOR: define internal void @test8(i32* nocapture readnone align 4 %a, i32* nocapture readnone align 4 %b, i32* nocapture readnone %c)
  ret void
}

declare void @test9_helper(i32* %A)
define void @test9_traversal(i1 %c, i32* align 4 %B, i32* align 8 %C) {
  %sel = select i1 %c, i32* %B, i32* %C
  call void @test9_helper(i32* %sel)
  ret void
}

; FIXME: This will work with an upcoming patch (D66618 or similar)
;             define align 32 i32* @test10a(i32* align 32 "no-capture-maybe-returned" %p)
; ATTRIBUTOR: define i32* @test10a(i32* nonnull align 32 dereferenceable(4) "no-capture-maybe-returned" %p)
define i32* @test10a(i32* align 32 %p) {
; ATTRIBUTOR: %l = load i32, i32* %p, align 32
  %l = load i32, i32* %p
  %c = icmp eq i32 %l, 0
  br i1 %c, label %t, label %f
t:
  %r = call i32* @test10a(i32* %p)
; FIXME: This will work with an upcoming patch (D66618 or similar)
;             store i32 1, i32* %r, align 32
; ATTRIBUTOR: store i32 1, i32* %r
  store i32 1, i32* %r
  %g0 = getelementptr i32, i32* %p, i32 8
  br label %e
f:
  %g1 = getelementptr i32, i32* %p, i32 8
; FIXME: This will work with an upcoming patch (D66618 or similar)
;             store i32 -1, i32* %g1, align 32
; ATTRIBUTOR: store i32 -1, i32* %g1
  store i32 -1, i32* %g1
  br label %e
e:
  %phi = phi i32* [%g0, %t], [%g1, %f]
  ret i32* %phi
}

; FIXME: This will work with an upcoming patch (D66618 or similar)
;             define align 32 i32* @test10b(i32* align 32 "no-capture-maybe-returned" %p)
; ATTRIBUTOR: define i32* @test10b(i32* nonnull align 32 dereferenceable(4) "no-capture-maybe-returned" %p)
define i32* @test10b(i32* align 32 %p) {
; ATTRIBUTOR: %l = load i32, i32* %p, align 32
  %l = load i32, i32* %p
  %c = icmp eq i32 %l, 0
  br i1 %c, label %t, label %f
t:
  %r = call i32* @test10b(i32* %p)
; FIXME: This will work with an upcoming patch (D66618 or similar)
;             store i32 1, i32* %r, align 32
; ATTRIBUTOR: store i32 1, i32* %r
  store i32 1, i32* %r
  %g0 = getelementptr i32, i32* %p, i32 8
  br label %e
f:
  %g1 = getelementptr i32, i32* %p, i32 -8
; FIXME: This will work with an upcoming patch (D66618 or similar)
;             store i32 -1, i32* %g1, align 32
; ATTRIBUTOR: store i32 -1, i32* %g1
  store i32 -1, i32* %g1
  br label %e
e:
  %phi = phi i32* [%g0, %t], [%g1, %f]
  ret i32* %phi
}

attributes #0 = { nounwind uwtable noinline }
attributes #1 = { uwtable noinline }
