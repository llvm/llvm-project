; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that we generate load instruction with (base + register offset << x)

; load word

define i32 @load_w(i32* nocapture %a, i32 %n, i32 %m) nounwind {
; CHECK: r{{[0-9]+}}{{ *}}={{ *}}memw(r{{[0-9]+}}{{ *}}+{{ *}}r{{[0-9]+}}{{ *}}<<{{ *}}#2)
entry:
  %tmp = add i32 %n, %m
  %scevgep9 = getelementptr i32, i32* %a, i32 %tmp
  %val = load i32, i32* %scevgep9, align 4
  ret i32 %val
}

; load unsigned half word

define i16 @load_uh(i16* nocapture %a, i32 %n, i32 %m) nounwind {
; CHECK: r{{[0-9]+}}{{ *}}={{ *}}memuh(r{{[0-9]+}}{{ *}}+{{ *}}r{{[0-9]+}}{{ *}}<<#1)
entry:
  %tmp = add i32 %n, %m
  %scevgep9 = getelementptr i16, i16* %a, i32 %tmp
  %val = load i16, i16* %scevgep9, align 2
  ret i16 %val
}

; load signed half word

define i32 @load_h(i16* nocapture %a, i32 %n, i32 %m) nounwind {
; CHECK: r{{[0-9]+}}{{ *}}={{ *}}memh(r{{[0-9]+}}{{ *}}+{{ *}}r{{[0-9]+}}{{ *}}<<#1)
entry:
  %tmp = add i32 %n, %m
  %scevgep9 = getelementptr i16, i16* %a, i32 %tmp
  %val = load i16, i16* %scevgep9, align 2
  %conv = sext i16 %val to i32
  ret i32 %conv
}

; load unsigned byte

define i8 @load_ub(i8* nocapture %a, i32 %n, i32 %m) nounwind {
; CHECK: r{{[0-9]+}}{{ *}}={{ *}}memub(r{{[0-9]+}}{{ *}}+{{ *}}r{{[0-9]+}}{{ *}}<<#0)
entry:
  %tmp = add i32 %n, %m
  %scevgep9 = getelementptr i8, i8* %a, i32 %tmp
  %val = load i8, i8* %scevgep9, align 1
  ret i8 %val
}

; load signed byte

define i32 @foo_2(i8* nocapture %a, i32 %n, i32 %m) nounwind {
; CHECK: r{{[0-9]+}}{{ *}}={{ *}}memb(r{{[0-9]+}}{{ *}}+{{ *}}r{{[0-9]+}}{{ *}}<<{{ *}}#0)
entry:
  %tmp = add i32 %n, %m
  %scevgep9 = getelementptr i8, i8* %a, i32 %tmp
  %val = load i8, i8* %scevgep9, align 1
  %conv = sext i8 %val to i32
  ret i32 %conv
}

; load doubleword

define i64 @load_d(i64* nocapture %a, i32 %n, i32 %m) nounwind {
; CHECK: r{{[0-9]+}}:{{[0-9]+}}{{ *}}={{ *}}memd(r{{[0-9]+}}{{ *}}+{{ *}}r{{[0-9]+}}{{ *}}<<{{ *}}#3)
entry:
  %tmp = add i32 %n, %m
  %scevgep9 = getelementptr i64, i64* %a, i32 %tmp
  %val = load i64, i64* %scevgep9, align 8
  ret i64 %val
}
