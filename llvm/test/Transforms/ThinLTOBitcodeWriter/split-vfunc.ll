; RUN: opt -thinlto-bc -o %t %s
; RUN: llvm-modextract -b -n 0 -o - %t | llvm-dis | FileCheck --check-prefix=M0 %s
; RUN: llvm-modextract -b -n 1 -o - %t | llvm-dis | FileCheck --check-prefix=M1 %s

; M0: @g = external constant [9 x i8*]{{$}}
; M1: @g = constant [9 x i8*]
@g = constant [9 x i8*] [
  i8* bitcast (i64 (i8*)* @ok1 to i8*),
  i8* bitcast (i64 (i8*, i64)* @ok2 to i8*),
  i8* bitcast (void (i8*)* @wrongtype1 to i8*),
  i8* bitcast (i128 (i8*)* @wrongtype2 to i8*),
  i8* bitcast (i64 ()* @wrongtype3 to i8*),
  i8* bitcast (i64 (i8*, i8*)* @wrongtype4 to i8*),
  i8* bitcast (i64 (i8*, i128)* @wrongtype5 to i8*),
  i8* bitcast (i64 (i8*)* @usesthis to i8*),
  i8* bitcast (i8 (i8*)* @reads to i8*)
], !type !0

; M0: define i64 @ok1
; M1: define available_externally i64 @ok1
define i64 @ok1(i8* %this) {
  ret i64 42
}

; M0: define i64 @ok2
; M1: define available_externally i64 @ok2
define i64 @ok2(i8* %this, i64 %arg) {
  %1 = tail call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %arg, i64 %arg)
  ret i64 %arg
}

; M1: declare { i64, i1 } @llvm.sadd.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.sadd.with.overflow.i64(i64, i64)

; M0: define void @wrongtype1
; M1: declare void @wrongtype1()
define void @wrongtype1(i8*) {
  ret void
}

; M0: define i128 @wrongtype2
; M1: declare void @wrongtype2()
define i128 @wrongtype2(i8*) {
  ret i128 0
}

; M0: define i64 @wrongtype3
; M1: declare void @wrongtype3()
define i64 @wrongtype3() {
  ret i64 0
}

; M0: define i64 @wrongtype4
; M1: declare void @wrongtype4()
define i64 @wrongtype4(i8*, i8*) {
  ret i64 0
}

; M0: define i64 @wrongtype5
; M1: declare void @wrongtype5()
define i64 @wrongtype5(i8*, i128) {
  ret i64 0
}

; M0: define i64 @usesthis
; M1: declare void @usesthis()
define i64 @usesthis(i8* %this) {
  %i = ptrtoint i8* %this to i64
  ret i64 %i
}

; M0: define i8 @reads
; M1: declare void @reads()
define i8 @reads(i8* %this) {
  %l = load i8, i8* %this
  ret i8 %l
}

!0 = !{i32 0, !"typeid"}
