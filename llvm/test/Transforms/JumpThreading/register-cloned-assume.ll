; RUN: opt -disable-output -passes='print<assumptions>,jump-threading,print<assumptions>' < %s 2>&1 | FileCheck %s

; Jump threading copies the assume in %merge into a threaded block. The copy
; must be registered with the assumption cache, which is already populated.
define void @thread_assume(i1 %c, ptr %p) {
; CHECK:      Cached assumptions for function: thread_assume
; CHECK-NEXT:   [ "nonnull"(ptr %p) ]
; CHECK-NEXT: Cached assumptions for function: thread_assume
; CHECK-NEXT:   [ "nonnull"(ptr %p) ]
; CHECK-NEXT:   [ "nonnull"(ptr %p) ]
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
