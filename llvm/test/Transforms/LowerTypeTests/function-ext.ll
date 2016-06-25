; RUN: opt -S -lowertypetests < %s | FileCheck %s

; Tests that we correctly handle external references, including the case where
; all functions in a bitset are external references.

target triple = "x86_64-unknown-linux-gnu"

declare !type !0 void @foo()

; CHECK: @[[JT:.*]] = private constant [1 x <{ i8, i32, i8, i8, i8 }>] [<{ i8, i32, i8, i8, i8 }> <{ i8 -23, i32 trunc (i64 sub (i64 sub (i64 ptrtoint (void ()* @foo to i64), i64 ptrtoint ([1 x <{ i8, i32, i8, i8, i8 }>]* @[[JT]] to i64)), i64 5) to i32), i8 -52, i8 -52, i8 -52 }>], section ".text"

define i1 @bar(i8* %ptr) {
  ; CHECK: icmp eq i64 {{.*}}, ptrtoint ([1 x <{ i8, i32, i8, i8, i8 }>]* @[[JT]] to i64)
  %p = call i1 @llvm.type.test(i8* %ptr, metadata !"void")
  ret i1 %p
}

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone

!0 = !{i64 0, !"void"}
