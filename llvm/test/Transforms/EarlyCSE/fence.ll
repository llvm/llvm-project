; RUN: opt -S -early-cse < %s | FileCheck %s
; RUN: opt < %s -S -basic-aa -early-cse-memssa | FileCheck %s
; NOTE: This file is testing the current implementation.  Some of
; the transforms used as negative tests below would be legal, but 
; only if reached through a chain of logic which EarlyCSE is incapable
; of performing.  To say it differently, this file tests a conservative
; version of the memory model.  If we want to extend EarlyCSE to be more
; aggressive in the future, we may need to relax some of the negative tests.

; We can value forward across the fence since we can (semantically) 
; reorder the following load before the fence.
define i32 @test(i32* %addr.i) {
; CHECK-LABEL: @test
; CHECK: store
; CHECK: fence
; CHECK-NOT: load
; CHECK: ret
  store i32 5, i32* %addr.i, align 4
  fence release
  %a = load i32, i32* %addr.i, align 4
  ret i32 %a
}

; Same as above
define i32 @test2(i32* noalias %addr.i, i32* noalias %otheraddr) {
; CHECK-LABEL: @test2
; CHECK: load
; CHECK: fence
; CHECK-NOT: load
; CHECK: ret
  %a = load i32, i32* %addr.i, align 4
  fence release
  %a2 = load i32, i32* %addr.i, align 4
  %res = sub i32 %a, %a2
  ret i32 %a
}

; We can not value forward across an acquire barrier since we might
; be syncronizing with another thread storing to the same variable
; followed by a release fence.  If this thread observed the release 
; had happened, we must present a consistent view of memory at the
; fence.  Note that it would be legal to reorder '%a' after the fence
; and then remove '%a2'.  The current implementation doesn't know how
; to do this, but if it learned, this test will need revised.
define i32 @test3(i32* noalias %addr.i, i32* noalias %otheraddr) {
; CHECK-LABEL: @test3
; CHECK: load
; CHECK: fence
; CHECK: load
; CHECK: sub
; CHECK: ret
  %a = load i32, i32* %addr.i, align 4
  fence acquire
  %a2 = load i32, i32* %addr.i, align 4
  %res = sub i32 %a, %a2
  ret i32 %res
}

; We can not dead store eliminate accross the fence.  We could in 
; principal reorder the second store above the fence and then DSE either
; store, but this is beyond the simple last-store DSE which EarlyCSE
; implements.
define void @test4(i32* %addr.i) {
; CHECK-LABEL: @test4
; CHECK: store
; CHECK: fence
; CHECK: store
; CHECK: ret
  store i32 5, i32* %addr.i, align 4
  fence release
  store i32 5, i32* %addr.i, align 4
  ret void
}

; We *could* DSE across this fence, but don't.  No other thread can
; observe the order of the acquire fence and the store.
define void @test5(i32* %addr.i) {
; CHECK-LABEL: @test5
; CHECK: store
; CHECK: fence
; CHECK: store
; CHECK: ret
  store i32 5, i32* %addr.i, align 4
  fence acquire
  store i32 5, i32* %addr.i, align 4
  ret void
}
