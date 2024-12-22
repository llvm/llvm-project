; RUN: opt -passes=simple-loop-unswitch -enable-nontrivial-unswitch -verify-memoryssa -S %s | FileCheck %s
; REQUIRES: asserts

target triple = "x86_64-unknown-linux-gnu"

declare void @foo()

; In Test1, there are no definitions. MemorySSA updates insert trivial phis and remove them.
; Verify all are removed, considering the SLU pass leaves unreachable blocks hanging when the MSSA updates are done.
; CHECK-LABEL: @Test1
define void @Test1(i32) {
header:
  br label %outer

outer.loopexit.split:                             ; preds = %continue
  br label %outer.loopexit

outer.loopexit:                                   ; preds = %outer.loopexit.split.us, %outer.loopexit.split
  br label %outer

outer:                                            ; preds = %outer.loopexit, %header
  br i1 false, label %outer.split.us, label %outer.split

outer.split.us:                                   ; preds = %outer
  br label %inner.us

inner.us:                                         ; preds = %continue.us, %outer.split.us
  br label %overflow.us

overflow.us:                                      ; preds = %inner.us
  br label %continue.us

continue.us:                                      ; preds = %overflow.us
  br i1 true, label %outer.loopexit.split.us, label %inner.us

outer.loopexit.split.us:                          ; preds = %continue.us
  br label %outer.loopexit

outer.split:                                      ; preds = %outer
  br label %inner

inner:                                            ; preds = %continue, %outer.split
  br label %switchme

switchme:                                         ; preds = %inner
  switch i32 %0, label %continue [
    i32 88, label %go_out
    i32 99, label %case2
  ]

case2:                                            ; preds = %switchme
  br label %continue

continue:                                         ; preds = %case2, %switchme
  br i1 true, label %outer.loopexit.split, label %inner

go_out:                                           ; preds = %switchme
  unreachable
}

; In Test2 there is a single def (call to foo). There are already Phis in place that are cloned when unswitching.
; Ensure MemorySSA remains correct. Due to SLU's pruned cloning, continue.us2 becomes unreachable, with an empty Phi that is later cleaned.
; CHECK-LABEL: @Test2
define void @Test2(i32) {
header:
  br label %outer

outer.loopexit.split:                             ; preds = %continue
  br label %outer.loopexit

outer.loopexit:                                   ; preds = %outer.loopexit.split.us, %outer.loopexit.split
  br label %outer

outer:                                            ; preds = %outer.loopexit, %header
  br i1 false, label %outer.split.us, label %outer.split

outer.split.us:                                   ; preds = %outer
  br label %inner.us

inner.us:                                         ; preds = %continue.us, %outer.split.us
  br label %overflow.us

overflow.us:                                      ; preds = %inner.us
  br label %continue.us

continue.us:                                      ; preds = %overflow.us
  br i1 true, label %outer.loopexit.split.us, label %inner.us

outer.loopexit.split.us:                          ; preds = %continue.us
  br label %outer.loopexit

outer.split:                                      ; preds = %outer
  br label %inner

inner:                                            ; preds = %continue, %outer.split
  br label %switchme

switchme:                                         ; preds = %inner
  switch i32 %0, label %continue [
    i32 88, label %go_out
    i32 99, label %case2
  ]

case2:                                            ; preds = %switchme
  call void @foo()
  br label %continue

continue:                                         ; preds = %case2, %switchme
  br i1 true, label %outer.loopexit.split, label %inner

go_out:                                           ; preds = %switchme
  unreachable
}
