; RUN: opt  -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract  -b -n 0 -o - %t | llvm-dis  | FileCheck --check-prefix=M0 %s
; RUN: llvm-modextract  -b -n 1 -o - %t | llvm-dis  | FileCheck --check-prefix=M1 %s

; M0: @g = external constant [10 x ptr]{{$}}
; M1: @g = constant [10 x ptr]
@g = constant [10 x ptr] [
  ptr @ok1,
  ptr @ok2,
  ptr @wrongtype1,
  ptr @wrongtype2,
  ptr @wrongtype3,
  ptr @wrongtype4,
  ptr @wrongtype5,
  ptr @usesthis,
  ptr @reads,
  ptr @attributedFunc
], !type !0

; M0: define i64 @ok1
; M1: define available_externally i64 @ok1
define i64 @ok1(ptr %this) {
  ret i64 42
}

; M0: define i64 @ok2
; M1: define available_externally i64 @ok2
define i64 @ok2(ptr %this, i64 %arg) {
  %1 = tail call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %arg, i64 %arg)
  ret i64 %arg
}

; M1: declare { i64, i1 } @llvm.sadd.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.sadd.with.overflow.i64(i64, i64)

; M0: define void @wrongtype1
; M1: declare void @wrongtype1()
define void @wrongtype1(ptr) {
  ret void
}

; M0: define i128 @wrongtype2
; M1: declare void @wrongtype2()
define i128 @wrongtype2(ptr) {
  ret i128 0
}

; M0: define i64 @wrongtype3
; M1: declare void @wrongtype3()
define i64 @wrongtype3() {
  ret i64 0
}

; M0: define i64 @wrongtype4
; M1: declare void @wrongtype4()
define i64 @wrongtype4(ptr, ptr) {
  ret i64 0
}

; M0: define i64 @wrongtype5
; M1: declare void @wrongtype5()
define i64 @wrongtype5(ptr, i128) {
  ret i64 0
}

; M0: define i64 @usesthis
; M1: declare void @usesthis()
define i64 @usesthis(ptr %this) {
  %i = ptrtoint ptr %this to i64
  ret i64 %i
}

; M0: define i8 @reads
; M1: declare void @reads()
define i8 @reads(ptr %this) {
  %l = load i8, ptr %this
  ret i8 %l
}

; Check function attributes are copied over splitted module
; M0: declare dso_local noundef ptr @attributedFunc(ptr noalias, i8 zeroext) unnamed_addr #[[ATTR0:[0-9]+]]
; M1: declare dso_local void @attributedFunc() unnamed_addr #[[ATTR1:[0-9]+]]
declare dso_local noundef ptr @attributedFunc(ptr noalias, i8 zeroext) unnamed_addr alwaysinline willreturn
; M0: attributes #[[ATTR0]] = { alwaysinline willreturn }
; M1: attributes #[[ATTR1]] = { alwaysinline willreturn }

!0 = !{i32 0, !"typeid"}
