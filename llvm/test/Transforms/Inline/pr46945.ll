; RUN: opt %s -o - -S -passes=always-inliner-wrapper | FileCheck %s
; RUN: opt %s -o - -S -passes='default<O2>' | FileCheck %s
; RUN: opt %s -o - -S -passes=inliner-wrapper | FileCheck %s -check-prefix=BASELINE

; In the baseline case, a will be first inlined into b, which makes c recursive,
; and, thus, un-inlinable. We need a baseline case to make sure intra-SCC order
; is as expected: b first, then a.

; BASELINE: call void @b()
; CHECK-NOT: call void @b()
define void @b() alwaysinline {
entry:
  br label %for.cond

for.cond:
  call void @a()
  br label %for.cond
}

define void @a() {
entry:
  call void @b()
  ret void
}
