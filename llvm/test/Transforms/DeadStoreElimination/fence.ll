; RUN: opt -S -passes=dse < %s | FileCheck %s

; We conservative choose to prevent dead store elimination
; across release or stronger fences.  It's not required 
; (since the must still be a race on %addd.i), but
; it is conservatively correct.  A legal optimization
; could hoist the second store above the fence, and then
; DSE one of them.
define void @test1(ptr %addr.i) {
; CHECK-LABEL: @test1
; CHECK: store i32 5
; CHECK: fence
; CHECK: store i32 5
; CHECK: ret
  store i32 5, ptr %addr.i, align 4
  fence release
  store i32 5, ptr %addr.i, align 4
  ret void
}

; Same as previous, but with different values.  If we ever optimize 
; this more aggressively, this allows us to check that the correct
; store is retained (the 'i32 1' store in this case)
define void @test1b(ptr %addr.i) {
; CHECK-LABEL: @test1b
; CHECK: store i32 42
; CHECK: fence release
; CHECK: store i32 1
; CHECK: ret
  store i32 42, ptr %addr.i, align 4
  fence release
  store i32 1, ptr %addr.i, align 4
  ret void
}

; We *could* DSE across this fence, but don't.  No other thread can
; observe the order of the acquire fence and the store.
define void @test2(ptr %addr.i) {
; CHECK-LABEL: @test2
; CHECK: store
; CHECK: fence
; CHECK: store
; CHECK: ret
  store i32 5, ptr %addr.i, align 4
  fence acquire
  store i32 5, ptr %addr.i, align 4
  ret void
}
