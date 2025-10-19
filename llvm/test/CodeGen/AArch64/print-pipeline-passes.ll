; RUN: opt -mtriple=aarch64 -S -passes='default<O0>' -print-pipeline-passes < %s | FileCheck --check-prefix=O0 %s
; RUN: opt -mtriple=aarch64 -S -passes='default<O2>' -print-pipeline-passes < %s | FileCheck %s

; CHECK: loop-idiom-vectorize
; O0: {{^}}function(ee-instrument<>),always-inline,coro-cond(coro-early,cgscc(coro-split),coro-cleanup,globaldce),function(annotation-remarks),verify,print{{$}}

define void @foo() {
entry:
  ret void
}
