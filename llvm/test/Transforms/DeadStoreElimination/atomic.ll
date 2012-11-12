; RUN: opt -basicaa -dse -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

; Sanity tests for atomic stores.
; Note that it turns out essentially every transformation DSE does is legal on
; atomic ops, just some transformations are not allowed across them. 

@x = common global i32 0, align 4
@y = common global i32 0, align 4

declare void @randomop(i32*)

; DSE across unordered store (allowed)
define void @test1()  nounwind uwtable ssp {
; CHECK: test1
; CHECK-NOT: store i32 0
; CHECK: store i32 1
entry:
  store i32 0, i32* @x
  store atomic i32 0, i32* @y unordered, align 4
  store i32 1, i32* @x
  ret void
}

; DSE across seq_cst load (allowed in theory; not implemented ATM)
define i32 @test2()  nounwind uwtable ssp {
; CHECK: test2
; CHECK: store i32 0
; CHECK: store i32 1
entry:
  store i32 0, i32* @x
  %x = load atomic i32* @y seq_cst, align 4
  store i32 1, i32* @x
  ret i32 %x
}

; DSE across seq_cst store (store before atomic store must not be removed)
define void @test3()  nounwind uwtable ssp {
; CHECK: test3
; CHECK: store i32
; CHECK: store atomic i32 2
entry:
  store i32 0, i32* @x
  store atomic i32 2, i32* @y seq_cst, align 4
  store i32 1, i32* @x
  ret void
}

; DSE remove unordered store (allowed)
define void @test4()  nounwind uwtable ssp {
; CHECK: test4
; CHECK-NOT: store atomic
; CHECK: store i32 1
entry:
  store atomic i32 0, i32* @x unordered, align 4
  store i32 1, i32* @x
  ret void
}

; DSE unordered store overwriting non-atomic store (allowed)
define void @test5()  nounwind uwtable ssp {
; CHECK: test5
; CHECK: store atomic i32 1
entry:
  store i32 0, i32* @x
  store atomic i32 1, i32* @x unordered, align 4
  ret void
}

; DSE no-op unordered atomic store (allowed)
define void @test6()  nounwind uwtable ssp {
; CHECK: test6
; CHECK-NOT: store
; CHECK: ret void
entry:
  %x = load atomic i32* @x unordered, align 4
  store atomic i32 %x, i32* @x unordered, align 4
  ret void
}

; DSE seq_cst store (be conservative; DSE doesn't have infrastructure
; to reason about atomic operations).
define void @test7()  nounwind uwtable ssp {
; CHECK: test7
; CHECK: store atomic 
entry:
  %a = alloca i32
  store atomic i32 0, i32* %a seq_cst, align 4
  ret void
}

; DSE and seq_cst load (be conservative; DSE doesn't have infrastructure
; to reason about atomic operations).
define i32 @test8()  nounwind uwtable ssp {
; CHECK: test8
; CHECK: store
; CHECK: load atomic 
entry:
  %a = alloca i32
  call void @randomop(i32* %a)
  store i32 0, i32* %a, align 4
  %x = load atomic i32* @x seq_cst, align 4
  ret i32 %x
}

