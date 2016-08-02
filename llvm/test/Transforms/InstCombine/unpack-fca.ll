; RUN: opt -instcombine -S < %s | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%A__vtbl = type { i8*, i32 (%A*)* }
%A = type { %A__vtbl* }
%B = type { i8*, i64 }

@A__vtblZ = constant %A__vtbl { i8* null, i32 (%A*)* @A.foo }

declare i32 @A.foo(%A* nocapture %this)

define void @storeA(%A* %a.ptr) {
; CHECK-LABEL: storeA
; CHECK-NEXT: [[GEP:%[a-z0-9\.]+]] = getelementptr inbounds %A, %A* %a.ptr, i64 0, i32 0
; CHECK-NEXT: store %A__vtbl* @A__vtblZ, %A__vtbl** [[GEP]], align 8
; CHECK-NEXT: ret void
  store %A { %A__vtbl* @A__vtblZ }, %A* %a.ptr, align 8
  ret void
}

define void @storeB(%B* %b.ptr) {
; CHECK-LABEL: storeB
; CHECK-NEXT: [[GEP1:%[a-z0-9\.]+]] = getelementptr inbounds %B, %B* %b.ptr, i64 0, i32 0
; CHECK-NEXT: store i8* null, i8** [[GEP1]], align 8
; CHECK-NEXT: [[GEP2:%[a-z0-9\.]+]] = getelementptr inbounds %B, %B* %b.ptr, i64 0, i32 1
; CHECK-NEXT: store i64 42, i64* [[GEP2]], align 8
; CHECK-NEXT: ret void
  store %B { i8* null, i64 42 }, %B* %b.ptr, align 8
  ret void
}

define void @storeStructOfA({ %A }* %sa.ptr) {
; CHECK-LABEL: storeStructOfA
; CHECK-NEXT: [[GEP:%[a-z0-9\.]+]] = getelementptr inbounds { %A }, { %A }* %sa.ptr, i64 0, i32 0, i32 0
; CHECK-NEXT: store %A__vtbl* @A__vtblZ, %A__vtbl** [[GEP]], align 8
; CHECK-NEXT: ret void
  store { %A } { %A { %A__vtbl* @A__vtblZ } }, { %A }* %sa.ptr, align 8
  ret void
}

define void @storeArrayOfA([1 x %A]* %aa.ptr) {
; CHECK-LABEL: storeArrayOfA
; CHECK-NEXT: [[GEP:%[a-z0-9\.]+]] = getelementptr inbounds [1 x %A], [1 x %A]* %aa.ptr, i64 0, i64 0, i32 0
; CHECK-NEXT: store %A__vtbl* @A__vtblZ, %A__vtbl** [[GEP]], align 8
; CHECK-NEXT: ret void
  store [1 x %A] [%A { %A__vtbl* @A__vtblZ }], [1 x %A]* %aa.ptr, align 8
  ret void
}

define void @storeStructOfArrayOfA({ [1 x %A] }* %saa.ptr) {
; CHECK-LABEL: storeStructOfArrayOfA
; CHECK-NEXT: [[GEP:%[a-z0-9\.]+]] = getelementptr inbounds { [1 x %A] }, { [1 x %A] }* %saa.ptr, i64 0, i32 0, i64 0, i32 0
; CHECK-NEXT: store %A__vtbl* @A__vtblZ, %A__vtbl** [[GEP]], align 8
; CHECK-NEXT: ret void
  store { [1 x %A] } { [1 x %A] [%A { %A__vtbl* @A__vtblZ }] }, { [1 x %A] }* %saa.ptr, align 8
  ret void
}

define %A @loadA(%A* %a.ptr) {
; CHECK-LABEL: loadA
; CHECK-NEXT: [[GEP:%[a-z0-9\.]+]] = getelementptr inbounds %A, %A* %a.ptr, i64 0, i32 0
; CHECK-NEXT: [[LOAD:%[a-z0-9\.]+]] = load %A__vtbl*, %A__vtbl** [[GEP]], align 8
; CHECK-NEXT: [[IV:%[a-z0-9\.]+]] = insertvalue %A undef, %A__vtbl* [[LOAD]], 0
; CHECK-NEXT: ret %A [[IV]]
  %1 = load %A, %A* %a.ptr, align 8
  ret %A %1
}

define %B @loadB(%B* %b.ptr) {
; CHECK-LABEL: loadB
; CHECK-NEXT: [[GEP1:%[a-z0-9\.]+]] = getelementptr inbounds %B, %B* %b.ptr, i64 0, i32 0
; CHECK-NEXT: [[LOAD1:%[a-z0-9\.]+]] = load i8*, i8** [[GEP1]], align 8
; CHECK-NEXT: [[IV1:%[a-z0-9\.]+]] = insertvalue %B undef, i8* [[LOAD1]], 0
; CHECK-NEXT: [[GEP2:%[a-z0-9\.]+]] = getelementptr inbounds %B, %B* %b.ptr, i64 0, i32 1
; CHECK-NEXT: [[LOAD2:%[a-z0-9\.]+]] = load i64, i64* [[GEP2]], align 8
; CHECK-NEXT: [[IV2:%[a-z0-9\.]+]] = insertvalue %B [[IV1]], i64 [[LOAD2]], 1
; CHECK-NEXT: ret %B [[IV2]]
  %1 = load %B, %B* %b.ptr, align 8
  ret %B %1
}

define { %A } @loadStructOfA({ %A }* %sa.ptr) {
; CHECK-LABEL: loadStructOfA
; CHECK-NEXT: [[GEP:%[a-z0-9\.]+]] = getelementptr inbounds { %A }, { %A }* %sa.ptr, i64 0, i32 0, i32 0
; CHECK-NEXT: [[LOAD:%[a-z0-9\.]+]] = load %A__vtbl*, %A__vtbl** [[GEP]], align 8
; CHECK-NEXT: [[IV1:%[a-z0-9\.]+]] = insertvalue %A undef, %A__vtbl* [[LOAD]], 0
; CHECK-NEXT: [[IV2:%[a-z0-9\.]+]] = insertvalue { %A } undef, %A [[IV1]], 0
; CHECK-NEXT: ret { %A } [[IV2]]
  %1 = load { %A }, { %A }* %sa.ptr, align 8
  ret { %A } %1
}

define [1 x %A] @loadArrayOfA([1 x %A]* %aa.ptr) {
; CHECK-LABEL: loadArrayOfA
; CHECK-NEXT: [[GEP:%[a-z0-9\.]+]] = getelementptr inbounds [1 x %A], [1 x %A]* %aa.ptr, i64 0, i64 0, i32 0
; CHECK-NEXT: [[LOAD:%[a-z0-9\.]+]] = load %A__vtbl*, %A__vtbl** [[GEP]], align 8
; CHECK-NEXT: [[IV1:%[a-z0-9\.]+]] = insertvalue %A undef, %A__vtbl* [[LOAD]], 0
; CHECK-NEXT: [[IV2:%[a-z0-9\.]+]] = insertvalue [1 x %A] undef, %A [[IV1]], 0
; CHECK-NEXT: ret [1 x %A] [[IV2]]
  %1 = load [1 x %A], [1 x %A]* %aa.ptr, align 8
  ret [1 x %A] %1
}

define { [1 x %A] } @loadStructOfArrayOfA({ [1 x %A] }* %saa.ptr) {
; CHECK-LABEL: loadStructOfArrayOfA
; CHECK-NEXT: [[GEP:%[a-z0-9\.]+]] = getelementptr inbounds { [1 x %A] }, { [1 x %A] }* %saa.ptr, i64 0, i32 0, i64 0, i32 0
; CHECK-NEXT: [[LOAD:%[a-z0-9\.]+]] = load %A__vtbl*, %A__vtbl** [[GEP]], align 8
; CHECK-NEXT: [[IV1:%[a-z0-9\.]+]] = insertvalue %A undef, %A__vtbl* [[LOAD]], 0
; CHECK-NEXT: [[IV2:%[a-z0-9\.]+]] = insertvalue [1 x %A] undef, %A [[IV1]], 0
; CHECK-NEXT: [[IV3:%[a-z0-9\.]+]] = insertvalue { [1 x %A] } undef, [1 x %A] [[IV2]], 0
; CHECK-NEXT: ret { [1 x %A] } [[IV3]]
  %1 = load { [1 x %A] }, { [1 x %A] }* %saa.ptr, align 8
  ret { [1 x %A] } %1
}

define { %A } @structOfA({ %A }* %sa.ptr) {
; CHECK-LABEL: structOfA
; CHECK-NEXT: [[GEP:%[a-z0-9\.]+]] = getelementptr inbounds { %A }, { %A }* %sa.ptr, i64 0, i32 0, i32 0
; CHECK-NEXT: store %A__vtbl* @A__vtblZ, %A__vtbl** [[GEP]], align 8
; CHECK-NEXT: ret { %A } { %A { %A__vtbl* @A__vtblZ } }
  store { %A } { %A { %A__vtbl* @A__vtblZ } }, { %A }* %sa.ptr, align 8
  %1 = load { %A }, { %A }* %sa.ptr, align 8
  ret { %A } %1
}

define %B @structB(%B* %b.ptr) {
; CHECK-LABEL: structB
; CHECK-NEXT: [[GEP1:%[a-z0-9\.]+]] = getelementptr inbounds %B, %B* %b.ptr, i64 0, i32 0
; CHECK-NEXT: store i8* null, i8** [[GEP1]], align 8
; CHECK-NEXT: [[GEP2:%[a-z0-9\.]+]] = getelementptr inbounds %B, %B* %b.ptr, i64 0, i32 1
; CHECK-NEXT: store i64 42, i64* [[GEP2]], align 8
; CHECK-NEXT: ret %B { i8* null, i64 42 }
  store %B { i8* null, i64 42 }, %B* %b.ptr, align 8
  %1 = load %B, %B* %b.ptr, align 8
  ret %B %1
}
