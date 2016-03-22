; RUN: opt -basicaa -loop-versioning -S < %s | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

define void @fill(i8** %ls1.20, i8** %ls2.21, i8* %cse3.22) {
; CHECK: bb1.lver.check:
; CHECK:   br i1 %memcheck.conflict, label %bb1.ph.lver.orig, label %bb1.ph
bb1.ph:
  %ls1.20.promoted = load i8*, i8** %ls1.20
  %ls2.21.promoted = load i8*, i8** %ls2.21
  br label %bb1

bb1:
  %_tmp302 = phi i8* [ %ls2.21.promoted, %bb1.ph ], [ %_tmp30, %bb1 ]
  %_tmp281 = phi i8* [ %ls1.20.promoted, %bb1.ph ], [ %_tmp28, %bb1 ]
  %_tmp14 = getelementptr i8, i8* %_tmp281, i16 -1
  %_tmp15 = load i8, i8* %_tmp14
  %add = add i8 %_tmp15, 1
  store i8 %add, i8* %_tmp281
  store i8 %add, i8* %_tmp302
  %_tmp28 = getelementptr i8, i8* %_tmp281, i16 1
  %_tmp30 = getelementptr i8, i8* %_tmp302, i16 1
  br i1 false, label %bb1, label %bb3.loopexit

bb3.loopexit:
  %_tmp30.lcssa = phi i8* [ %_tmp30, %bb1 ]
  %_tmp15.lcssa = phi i8 [ %_tmp15, %bb1 ]
  %_tmp28.lcssa = phi i8* [ %_tmp28, %bb1 ]
  store i8* %_tmp28.lcssa, i8** %ls1.20
  store i8 %_tmp15.lcssa, i8* %cse3.22
  store i8* %_tmp30.lcssa, i8** %ls2.21
  br label %bb3

bb3:
  ret void
}
