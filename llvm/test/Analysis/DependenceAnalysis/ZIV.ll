; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1 \
; RUN: | FileCheck %s
; RUN: opt < %s -analyze -basic-aa -da | FileCheck %s

; ModuleID = 'ZIV.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.6.0"


;;  A[n + 1] = 0;
;;  *B = A[1 + n];

define void @z0(i32* %A, i32* %B, i64 %n) nounwind uwtable ssp {
entry:
  %add = add i64 %n, 1
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %add
  store i32 0, i32* %arrayidx, align 4

; CHECK: da analyze - none!
; CHECK: da analyze - consistent flow [|<]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

  %add1 = add i64 %n, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %add1
  %0 = load i32, i32* %arrayidx2, align 4
  store i32 %0, i32* %B, align 4
  ret void
}


;;  A[n] = 0;
;;  *B = A[n + 1];

define void @z1(i32* %A, i32* %B, i64 %n) nounwind uwtable ssp {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %n
  store i32 0, i32* %arrayidx, align 4

; CHECK: da analyze - none!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

  %add = add i64 %n, 1
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i64 %add
  %0 = load i32, i32* %arrayidx1, align 4
  store i32 %0, i32* %B, align 4
  ret void
}


;;  A[n] = 0;
;;  *B = A[m];

define void @z2(i32* %A, i32* %B, i64 %n, i64 %m) nounwind uwtable ssp {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %n
  store i32 0, i32* %arrayidx, align 4

; CHECK: da analyze - none!
; CHECK: da analyze - flow [|<]!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!
; CHECK: da analyze - confused!
; CHECK: da analyze - none!

  %arrayidx1 = getelementptr inbounds i32, i32* %A, i64 %m
  %0 = load i32, i32* %arrayidx1, align 4
  store i32 %0, i32* %B, align 4
  ret void
}
