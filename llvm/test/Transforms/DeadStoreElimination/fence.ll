; RUN: opt -S -basic-aa -dse < %s | FileCheck %s

; We conservative choose to prevent dead store elimination
; across release or stronger fences.  It's not required 
; (since the must still be a race on %addd.i), but
; it is conservatively correct.  A legal optimization
; could hoist the second store above the fence, and then
; DSE one of them.
define void @test1(i32* %addr.i) {
; CHECK-LABEL: @test1
; CHECK: store i32 5
; CHECK: fence
; CHECK: store i32 5
; CHECK: ret
  store i32 5, i32* %addr.i, align 4
  fence release
  store i32 5, i32* %addr.i, align 4
  ret void
}

; Same as previous, but with different values.  If we ever optimize 
; this more aggressively, this allows us to check that the correct
; store is retained (the 'i32 1' store in this case)
define void @test1b(i32* %addr.i) {
; CHECK-LABEL: @test1b
; CHECK: store i32 42
; CHECK: fence release
; CHECK: store i32 1
; CHECK: ret
  store i32 42, i32* %addr.i, align 4
  fence release
  store i32 1, i32* %addr.i, align 4
  ret void
}

; We *could* DSE across this fence, but don't.  No other thread can
; observe the order of the acquire fence and the store.
define void @test2(i32* %addr.i) {
; CHECK-LABEL: @test2
; CHECK: store
; CHECK: fence
; CHECK: store
; CHECK: ret
  store i32 5, i32* %addr.i, align 4
  fence acquire
  store i32 5, i32* %addr.i, align 4
  ret void
}

; We DSE stack alloc'ed and byval locations, in the presence of fences.
; Fence does not make an otherwise thread local store visible.
; Right now the DSE in presence of fence is only done in end blocks (with no successors),
; but the same logic applies to other basic blocks as well.
; The store to %addr.i can be removed since it is a byval attribute
define void @test3(i32* byval %addr.i) {
; CHECK-LABEL: @test3
; CHECK-NOT: store
; CHECK: fence
; CHECK: ret
  store i32 5, i32* %addr.i, align 4
  fence release
  ret void
}

declare void @foo(i8* nocapture %p)

declare noalias i8* @malloc(i32)

; DSE of stores in locations allocated through library calls.
define void @test_nocapture() {
; CHECK-LABEL: @test_nocapture
; CHECK: malloc
; CHECK: foo
; CHECK-NOT: store
; CHECK: fence
  %m  =  call i8* @malloc(i32 24)
  call void @foo(i8* %m)
  store i8 4, i8* %m
  fence release
  ret void
}


; This is a full fence, but it does not make a thread local store visible.
; We can DSE the store in presence of the fence.
define void @fence_seq_cst() {
; CHECK-LABEL: @fence_seq_cst
; CHECK-NEXT: fence seq_cst
; CHECK-NEXT: ret void
  %P1 = alloca i32
  store i32 0, i32* %P1, align 4
  fence seq_cst
  store i32 4, i32* %P1, align 4
  ret void
}

