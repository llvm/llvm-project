; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7--linux-gnueabi"

; CHECK-LABEL: test_zext_sext_amounts255
; CHECK: NoAlias: i8* %a, i8* %b
define void @test_zext_sext_amounts255(ptr %mem) {
  %sext.1 = sext i8 255 to i16
  %sext.zext.1 = zext i16 %sext.1 to i64
  %sext.2 = sext i8 255 to i32
  %sext.zext.2 = zext i32 %sext.2 to i64
  %a = getelementptr inbounds i8, ptr %mem, i64 %sext.zext.1
  %b = getelementptr inbounds i8, ptr %mem, i64 %sext.zext.2
  load i8, ptr %a
  load i8, ptr %b
  ret void
}

; CHECK-LABEL: test_zext_sext_amounts
; CHECK: MayAlias: i8* %a, i8* %b
; %a and %b only PartialAlias as, although they're both zext(sext(%num)) they'll extend the sign by a different
; number of bits before zext-ing the remainder.
define void @test_zext_sext_amounts(ptr %mem, i8 %num) {
  %sext.1 = sext i8 %num to i16
  %sext.zext.1 = zext i16 %sext.1 to i64
  %sext.2 = sext i8 %num to i32
  %sext.zext.2 = zext i32 %sext.2 to i64
  %a = getelementptr inbounds i8, ptr %mem, i64 %sext.zext.1
  %b = getelementptr inbounds i8, ptr %mem, i64 %sext.zext.2
  load i8, ptr %a
  load i8, ptr %b
  ret void
}

; CHECK-LABEL: based_on_pr18068
; CHECK: NoAlias: i8* %a, i8* %b
; CHECK: NoAlias: i8* %a, i8* %c
define void @based_on_pr18068(i32 %loaded, ptr %mem) {
  %loaded.64 = zext i32 %loaded to i64
  %add1 = add i32 %loaded, -1 ; unsigned wraps unless %loaded == 0
  %add1.64 = zext i32 %add1 to i64 ; is zext(%loaded) always != zext(%loaded - 1)? Yes -> NoAlias
  %sub1 = sub i32 %loaded, 1 ; unsigned wraps iff %loaded == 0
  %sub1.64 = zext i32 %sub1 to i64 ; is zext(%loaded) always != zext(%loaded - 1)? Yes -> NoAlias
  %a = getelementptr inbounds i8, ptr %mem, i64 %loaded.64
  %b = getelementptr inbounds i8, ptr %mem, i64 %add1.64
  %c = getelementptr inbounds i8, ptr %mem, i64 %sub1.64
  load i8, ptr %a
  load i8, ptr %b
  load i8, ptr %c
  ret void
}

; CHECK-LABEL: test_path_dependence
; CHECK: MayAlias: i8* %a, i8* %b
; CHECK: MustAlias: i8* %a, i8* %c
; CHECK: MayAlias: i8* %a, i8* %d
define void @test_path_dependence(i16 %p, ptr %mem) {
  %p.minus1 = add i16 %p, -1 ; this will always unsigned-wrap, unless %p == 0
  %p.minus1.64 = zext i16 %p.minus1 to i64
  %p.64.again = add i64 %p.minus1.64, 1 ; either %p (if we wrapped) or 65536 (if we didn't)

  %p.nsw.nuw.minus1 = sub nsw nuw i16 %p, 1 ; as nuw we know %p >= 1, and as nsw %p <= 32767
  %p.nsw.nuw.minus1.64 = zext i16 %p.nsw.nuw.minus1 to i64
  %p.nsw.nuw.64.again = add nsw nuw i64 %p.nsw.nuw.minus1.64, 1 ; ...so always exactly %p

  %p.nsw.minus1 = sub nsw i16 %p, 1 ; only nsw, so can only guarantee %p != 0x1000
  %p.nsw.minus1.64 = zext i16 %p.nsw.minus1 to i64 ; when %p > 0x1000 (ie <= 0 as a signed number) then the zext will make this a huge positive number
  %p.nsw.64.again = add nsw i64 %p.nsw.minus1.64, 1 ; ...and so this is very much != %p

  %p.64 = zext i16 %p to i64
  %a = getelementptr inbounds i8, ptr %mem, i64 %p.64
  %b = getelementptr inbounds i8, ptr %mem, i64 %p.64.again
  %c = getelementptr inbounds i8, ptr %mem, i64 %p.nsw.nuw.64.again
  %d = getelementptr inbounds i8, ptr %mem, i64 %p.nsw.64.again
  load i8, ptr %a
  load i8, ptr %b
  load i8, ptr %c
  load i8, ptr %d
  ret void
}

; CHECK-LABEL: test_zext_sext_255
; CHECK: NoAlias: i8* %a, i8* %b
define void @test_zext_sext_255(ptr %mem) {
  %zext.255 = zext i8 255 to i16 ; 0x00FF
  %sext.255 = sext i8 255 to i16 ; 0xFFFF
  %zext.sext.255 = zext i16 %sext.255 to i32 ; 0x0000FFFF
  %sext.zext.255 = sext i16 %zext.255 to i32 ; 0x000000FF
  %zext.zext.sext.255 = zext i32 %zext.sext.255 to i64
  %zext.sext.zext.255 = zext i32 %sext.zext.255 to i64
  %a = getelementptr inbounds i8, ptr %mem, i64 %zext.zext.sext.255
  %b = getelementptr inbounds i8, ptr %mem, i64 %zext.sext.zext.255
  load i8, ptr %a
  load i8, ptr %b
  ret void
}

; CHECK-LABEL: test_zext_sext_num
; CHECK: MayAlias: i8* %a, i8* %b
; %a and %b NoAlias if %num == 255 (see @test_zext_sext_255), but %a and %b NoAlias for other values of %num (e.g. 0)
define void @test_zext_sext_num(ptr %mem, i8 %num) {
  %zext.num = zext i8 %num to i16
  %sext.num = sext i8 %num to i16
  %zext.sext.num = zext i16 %sext.num to i32
  %sext.zext.num = sext i16 %zext.num to i32
  %zext.zext.sext.num = zext i32 %zext.sext.num to i64
  %zext.sext.zext.num = zext i32 %sext.zext.num to i64
  %a = getelementptr inbounds i8, ptr %mem, i64 %zext.zext.sext.num
  %b = getelementptr inbounds i8, ptr %mem, i64 %zext.sext.zext.num
  load i8, ptr %a
  load i8, ptr %b
  ret void
}

; CHECK-LABEL: uncompressStream
; CHECK: MustAlias:  i8* %a, i8* %b
; CHECK: NoAlias:  i8* %a, i8* %c
define void @uncompressStream(ptr %mem) {
  %zext.255 = zext i8 255 to i32
  %sext.255 = sext i8 255 to i32
  %a = getelementptr inbounds i8, ptr %mem, i32 255
  %b = getelementptr inbounds i8, ptr %mem, i32 %zext.255
  %c = getelementptr inbounds i8, ptr %mem, i32 %sext.255
  load i8, ptr %a
  load i8, ptr %b
  load i8, ptr %c
  ret void
}

; CHECK-LABEL: constantOffsetHeuristic_i3_i32
; CHECK: NoAlias:  i32* %a, i32* %b
; CHECK: NoAlias:  i32* %a, i32* %c
; CHECK: NoAlias:  i32* %b, i32* %c
define void @constantOffsetHeuristic_i3_i32(ptr %mem, i3 %val) {
  %zext.plus.7 = add nsw i3 %val, 7
  %zext.plus.4 = add nsw i3 %val, 4
  %zext.val = zext i3 %val to i32
  %zext.4 = zext i3 %zext.plus.4 to i32
  %zext.7 = zext i3 %zext.plus.7 to i32
  %a = getelementptr inbounds i32, ptr %mem, i32 %zext.4
  %b = getelementptr inbounds i32, ptr %mem, i32 %zext.7
  %c = getelementptr inbounds i32, ptr %mem, i32 %zext.val
  load i32, ptr %a
  load i32, ptr %b
  load i32, ptr %c
  ret void
}

; CHECK-LABEL: constantOffsetHeuristic_i8_i32
; CHECK: NoAlias:  i32* %a, i32* %b
; CHECK: NoAlias:  i32* %a, i32* %c
; CHECK: NoAlias:  i32* %b, i32* %c
define void @constantOffsetHeuristic_i8_i32(ptr %mem, i8 %val) {
  %zext.plus.7 = add nsw i8 %val, 7
  %zext.plus.4 = add nsw i8 %val, 4
  %zext.val = zext i8 %val to i32
  %zext.4 = zext i8 %zext.plus.4 to i32
  %zext.7 = zext i8 %zext.plus.7 to i32
  %a = getelementptr inbounds i32, ptr %mem, i32 %zext.4
  %b = getelementptr inbounds i32, ptr %mem, i32 %zext.7
  %c = getelementptr inbounds i32, ptr %mem, i32 %zext.val
  load i32, ptr %a
  load i32, ptr %b
  load i32, ptr %c
  ret void
}

; CHECK-LABEL: constantOffsetHeuristic_i3_i8
; CHECK: MayAlias:  i32* %a.8, i32* %b.8
; CHECK: NoAlias:  i32* %a.8, i32* %c.8
; CHECK: MayAlias:  i32* %b.8, i32* %c.8
define void @constantOffsetHeuristic_i3_i8(ptr %mem, i3 %val) {
  %zext.plus.7 = add nsw i3 %val, 7
  %zext.plus.4 = add nsw i3 %val, 4
  %zext.val = zext i3 %val to i32
  %zext.4 = zext i3 %zext.plus.4 to i32
  %zext.7 = zext i3 %zext.plus.7 to i32
  %a.8 = getelementptr inbounds i8, ptr %mem, i32 %zext.4
  %b.8 = getelementptr inbounds i8, ptr %mem, i32 %zext.7
  %c.8 = getelementptr inbounds i8, ptr %mem, i32 %zext.val
  load i32, ptr %a.8
  load i32, ptr %b.8
  load i32, ptr %c.8
  ret void
}

; CHECK-LABEL: constantOffsetHeuristic_i8_i8
; CHECK: MayAlias:  i32* %a.8, i32* %b.8
; CHECK: NoAlias:  i32* %a.8, i32* %c.8
; CHECK: NoAlias:  i32* %b.8, i32* %c.8
define void @constantOffsetHeuristic_i8_i8(ptr %mem, i8 %val) {
  %zext.plus.7 = add nsw i8 %val, 7
  %zext.plus.4 = add nsw i8 %val, 4
  %zext.val = zext i8 %val to i32
  %zext.4 = zext i8 %zext.plus.4 to i32
  %zext.7 = zext i8 %zext.plus.7 to i32
  %a.8 = getelementptr inbounds i8, ptr %mem, i32 %zext.4
  %b.8 = getelementptr inbounds i8, ptr %mem, i32 %zext.7
  %c.8 = getelementptr inbounds i8, ptr %mem, i32 %zext.val
  load i32, ptr %a.8
  load i32, ptr %b.8
  load i32, ptr %c.8
  ret void
}

; CHECK-LABEL: different_large_bitwidths
; MayAlias: ptr %p1, ptr %p2
define void @different_large_bitwidths(ptr %a, i64 %i, i128 %j) {
  %p1 = getelementptr i8, ptr %a, i64 %i
  %p2 = getelementptr i8, ptr %a, i128 %j
  load i8, ptr %p1
  load i8, ptr %p2
  ret void
}
