; RUN: llc -mtriple=x86_64-apple-darwin8 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-pc-linux < %s | FileCheck %s

; FIXME: x32 doesn't know how to select this.  This isn't a regression, it never
; worked.
; RUNX: llc -mtriple=x86_64-pc-linux-gnux32 < %s | FileCheck -check-prefix=X32ABI %s

; This used to crash due to topological sorting issues in selection DAG.
define void @foo(ptr sret(i32) %agg.result, i32, i32, i32, i32, i32, ptr %pred) {
entry:
  call void %pred(i32 undef)
  ret void

; CHECK-LABEL: foo:
; CHECK: callq
; CHECK: movq {{.*}}, %rax
; CHECK: ret
}
