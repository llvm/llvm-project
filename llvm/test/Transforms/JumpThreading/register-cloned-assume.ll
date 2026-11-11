; RUN: opt -S -passes='require<assumptions>,jump-threading,drop-unnecessary-assumes' < %s | FileCheck %s

; Jump threading copies the assume in %merge into the threaded blocks. The
; copies must be registered with the assumption cache, or the cache-driven
; drop below would not see them.
define void @thread_assume(i1 %c, ptr %p) {
; CHECK-LABEL: define void @thread_assume(
; CHECK-NOT: call void @llvm.assume
; CHECK: ret void
entry:
  br i1 %c, label %a, label %b

a:
  call void @foo()
  br label %merge

b:
  call void @bar()
  br label %merge

merge:
  %phi = phi i1 [ true, %a ], [ false, %b ]
  call void @llvm.assume(i1 true) [ "nonnull"(ptr %p) ]
  br i1 %phi, label %t, label %f

t:
  call void @foo()
  ret void

f:
  call void @bar()
  ret void
}

declare void @foo()
declare void @bar()
