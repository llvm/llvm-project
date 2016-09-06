; RUN: opt -S -lowertypetests -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck --check-prefix=X64 %s
; RUN: opt -S -lowertypetests -mtriple=wasm32-unknown-unknown < %s | FileCheck --check-prefix=WASM32 %s

; Tests that we correctly handle bitsets containing 2 or more functions.

target datalayout = "e-p:64:64"

; X64: @[[JT:.*]] = private constant [2 x <{ i8, i32, i8, i8, i8 }>] [<{ i8, i32, i8, i8, i8 }> <{ i8 -23, i32 trunc (i64 sub (i64 sub (i64 ptrtoint (void ()* @[[FNAME:.*]] to i64), i64 ptrtoint ([2 x <{ i8, i32, i8, i8, i8 }>]* @[[JT]] to i64)), i64 5) to i32), i8 -52, i8 -52, i8 -52 }>, <{ i8, i32, i8, i8, i8 }> <{ i8 -23, i32 trunc (i64 sub (i64 sub (i64 ptrtoint (void ()* @[[GNAME:.*]] to i64), i64 ptrtoint ([2 x <{ i8, i32, i8, i8, i8 }>]* @[[JT]] to i64)), i64 13) to i32), i8 -52, i8 -52, i8 -52 }>], section ".text"
; WASM32: private constant [0 x i8] zeroinitializer
@0 = private unnamed_addr constant [2 x void (...)*] [void (...)* bitcast (void ()* @f to void (...)*), void (...)* bitcast (void ()* @g to void (...)*)], align 16

; X64: @f = alias void (), bitcast ([2 x <{ i8, i32, i8, i8, i8 }>]* @[[JT]] to void ()*)
; X64: @g = alias void (), bitcast (<{ i8, i32, i8, i8, i8 }>* getelementptr inbounds ([2 x <{ i8, i32, i8, i8, i8 }>], [2 x <{ i8, i32, i8, i8, i8 }>]* @[[JT]], i64 0, i64 1) to void ()*)

; X64: define private void @[[FNAME]]()
; WASM32: define void @f() !type !{{[0-9]+}} !wasm.index ![[I0:[0-9]+]]
define void @f() !type !0 {
  ret void
}

; X64: define private void @[[GNAME]]()
; WASM32: define void @g() !type !{{[0-9]+}} !wasm.index ![[I1:[0-9]+]]
define void @g() !type !0 {
  ret void
}

!0 = !{i32 0, !"typeid1"}

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone

define i1 @foo(i8* %p) {
  ; X64: sub i64 {{.*}}, ptrtoint ([2 x <{ i8, i32, i8, i8, i8 }>]* @[[JT]] to i64)
  ; WASM32: sub i64 {{.*}}, 1
  ; WASM32: icmp ult i64 {{.*}}, 2
  %x = call i1 @llvm.type.test(i8* %p, metadata !"typeid1")
  ret i1 %x
}

; WASM32: ![[I0]] = !{i64 1}
; WASM32: ![[I1]] = !{i64 2}
