; RUN: opt -S -lowertypetests -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck --check-prefix=X64 %s
; RUN: opt -S -lowertypetests -mtriple=wasm32-unknown-unknown < %s | FileCheck --check-prefix=WASM32 %s

; Tests that we correctly handle bitsets with disjoint call target sets.

target datalayout = "e-p:64:64"

; X64: @[[JT0:.*]] = private constant [1 x <{ i8, i32, i8, i8, i8 }>] [<{ i8, i32, i8, i8, i8 }> <{ i8 -23, i32 trunc (i64 sub (i64 sub (i64 ptrtoint (void ()* @[[FNAME:.*]] to i64), i64 ptrtoint ([1 x <{ i8, i32, i8, i8, i8 }>]* @[[JT0]] to i64)), i64 5) to i32), i8 -52, i8 -52, i8 -52 }>], section ".text"
; X64: @[[JT1:.*]] = private constant [1 x <{ i8, i32, i8, i8, i8 }>] [<{ i8, i32, i8, i8, i8 }> <{ i8 -23, i32 trunc (i64 sub (i64 sub (i64 ptrtoint (void ()* @[[GNAME:.*]] to i64), i64 ptrtoint ([1 x <{ i8, i32, i8, i8, i8 }>]* @[[JT1]] to i64)), i64 5) to i32), i8 -52, i8 -52, i8 -52 }>], section ".text"
; WASM32: private constant [0 x i8] zeroinitializer
@0 = private unnamed_addr constant [2 x void ()*] [void ()* @f, void ()* @g], align 16

; X64: @f = alias void (), bitcast ([1 x <{ i8, i32, i8, i8, i8 }>]* @[[JT0]] to void ()*)
; X64: @g = alias void (), bitcast ([1 x <{ i8, i32, i8, i8, i8 }>]* @[[JT1]] to void ()*)

; X64: define private void @[[FNAME]]()
; WASM32: define void @f() !type !{{[0-9]+}} !wasm.index ![[I0:[0-9]+]]
define void @f() !type !0 {
  ret void
}

; X64: define private void @[[GNAME]]()
; WASM32: define void @g() !type !{{[0-9]+}} !wasm.index ![[I1:[0-9]+]]
define void @g() !type !1 {
  ret void
}

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone

define i1 @foo(i8* %p) {
  ; X64: icmp eq i64 {{.*}}, ptrtoint ([1 x <{ i8, i32, i8, i8, i8 }>]* @[[JT0]] to i64)
  ; WASM32: icmp eq i64 {{.*}}, 1
  %x = call i1 @llvm.type.test(i8* %p, metadata !"typeid1")
  ; X64: icmp eq i64 {{.*}}, ptrtoint ([1 x <{ i8, i32, i8, i8, i8 }>]* @[[JT1]] to i64)
  ; WASM32: icmp eq i64 {{.*}}, 2
  %y = call i1 @llvm.type.test(i8* %p, metadata !"typeid2")
  %z = add i1 %x, %y
  ret i1 %z
}

; WASM32: ![[I0]] = !{i64 1}
; WASM32: ![[I1]] = !{i64 2}
