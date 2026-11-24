; RUN: opt -passes='require<assumptions>,jump-threading,print<assumptions>' -disable-output %s 2>&1 | FileCheck %s

; The assume in %merge is cloned into both predecessors when threading the
; branch on %c. The clones must be registered in the computed assumption cache.

; CHECK-LABEL: Cached assumptions for function: thread_assume
; CHECK-NEXT: {{^}}  i1 %x{{$}}
; CHECK-NEXT: {{^}}  i1 %x{{$}}
define void @thread_assume(i1 %c, i1 %x) {
entry:
  br i1 %c, label %if, label %merge

if:
  %v = load i32, ptr null, align 4
  br label %merge

merge:
  call void @llvm.assume(i1 %x)
  br i1 %c, label %a, label %b

a:
  ret void

b:
  ret void
}
