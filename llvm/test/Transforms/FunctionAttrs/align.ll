; RUN: opt -attributor -attributor-disable=false -S < %s | FileCheck %s --check-prefix=ATTRIBUTOR

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Test cases specifically designed for "align" attribute.
; We use FIXME's to indicate problems and missing attributes.


; TEST 1
; ATTRIBUTOR: define align 8 i32* @test1(i32* returned align 8)
define i32* @test1(i32* align 8) #0 {
  ret i32* %0
}

; TEST 2
; ATTRIBUTOR: define i32* @test2(i32* returned)
define i32* @test2(i32*) #0 {
  ret i32* %0
}

; TEST 3
; ATTRIBUTOR: define align 4 i32* @test3(i32* align 8, i32* align 4, i1)
define i32* @test3(i32* align 8, i32* align 4, i1) #0 {
  %ret = select i1 %2, i32* %0, i32* %1
  ret i32* %ret
}

; TEST 4
; ATTRIBUTOR: define align 32 i32* @test4(i32* align 32, i32* align 32, i1)
define i32* @test4(i32* align 32, i32* align 32, i1) #0 {
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
define internal i8* @f1(i8* readnone) local_unnamed_addr #0 {
; ATTRIBUTOR: define internal nonnull align 8 i8* @f1(i8* nonnull readnone align 8)
  %2 = icmp eq i8* %0, null
  br i1 %2, label %3, label %5

; <label>:3:                                      ; preds = %1
; ATTRIBUTOR: %4 = tail call i8* @f2(i8* nonnull align 8 @a1)
  %4 = tail call i8* @f2(i8* nonnull @a1)
  br label %5

; <label>:5:                                      ; preds = %1, %3
  %6 = phi i8* [ %4, %3 ], [ %0, %1 ]
  ret i8* %6
}

; Function Attrs: nounwind readnone ssp uwtable
define internal i8* @f2(i8* readnone) local_unnamed_addr #0 {
; ATTRIBUTOR: define internal nonnull align 8 i8* @f2(i8* nonnull readnone align 8)
  %2 = icmp eq i8* %0, null
  br i1 %2, label %5, label %3

; <label>:3:                                      ; preds = %1

; ATTRIBUTOR: %4 = tail call i8* @f1(i8* nonnull align 8 %0)
  %4 = tail call i8* @f1(i8* nonnull %0)
  br label %7

; <label>:5:                                      ; preds = %1
; ATTRIBUTOR: %6 = tail call i8* @f3(i8* nonnull align 16 @a2)
  %6 = tail call i8* @f3(i8* nonnull @a2)
  br label %7

; <label>:7:                                      ; preds = %5, %3
  %8 = phi i8* [ %4, %3 ], [ %6, %5 ]
  ret i8* %8
}

; Function Attrs: nounwind readnone ssp uwtable
define internal i8* @f3(i8* readnone) local_unnamed_addr #0 {
; ATTRIBUTOR: define internal nonnull align 8 i8* @f3(i8* nonnull readnone align 16)
  %2 = icmp eq i8* %0, null
  br i1 %2, label %3, label %5

; <label>:3:                                      ; preds = %1
; ATTRIBUTOR: %4 = tail call i8* @f1(i8* nonnull align 16 @a2)
  %4 = tail call i8* @f1(i8* nonnull @a2)
  br label %5

; <label>:5:                                      ; preds = %1, %3
  %6 = phi i8* [ %4, %3 ], [ @a1, %1 ]
  ret i8* %6
}

; TEST 7
; Better than IR information
; ATTRIBUTOR: define align 32 i32* @test7(i32* returned align 32 %p)
define align 4 i32* @test7(i32* align 32 %p) #0 {
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
; ATTRIBUTOR: define internal void @test8(i32* align 4 %a, i32* align 4 %b, i32* %c)
  ret void
}


attributes #0 = { nounwind uwtable noinline }
attributes #1 = { uwtable noinline }
