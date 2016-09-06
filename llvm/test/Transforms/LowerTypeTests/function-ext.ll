; RUN: opt -S -lowertypetests -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck --check-prefix=X64 %s
; RUN: opt -S -lowertypetests -mtriple=wasm32-unknown-unknown < %s | FileCheck --check-prefix=WASM32 %s

; Tests that we correctly handle external references, including the case where
; all functions in a bitset are external references.

; X64: @[[JT:.*]] = private constant [1 x <{ i8, i32, i8, i8, i8 }>] [<{ i8, i32, i8, i8, i8 }> <{ i8 -23, i32 trunc (i64 sub (i64 sub (i64 ptrtoint (void ()* @foo to i64), i64 ptrtoint ([1 x <{ i8, i32, i8, i8, i8 }>]* @[[JT]] to i64)), i64 5) to i32), i8 -52, i8 -52, i8 -52 }>], section ".text"
; WASM32: private constant [0 x i8] zeroinitializer

; WASM32: declare !type !{{[0-9]+}} void @foo()
declare !type !0 void @foo()

define i1 @bar(i8* %ptr) {
  ; X64: icmp eq i64 {{.*}}, ptrtoint ([1 x <{ i8, i32, i8, i8, i8 }>]* @[[JT]] to i64)
  ; WASM32: sub i64 {{.*}}, 0
  ; WASM32: icmp ult i64 {{.*}}, 1
  %p = call i1 @llvm.type.test(i8* %ptr, metadata !"void")
  ret i1 %p
}

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone

!0 = !{i64 0, !"void"}
; WASM-NOT: !{i64 0}
; WASM-NOT: !{i64 1}
