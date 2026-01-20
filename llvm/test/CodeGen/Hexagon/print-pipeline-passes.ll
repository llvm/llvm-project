; RUN: opt -mtriple=hexagon -S -passes='default<O0>' -print-pipeline-passes < %s 2>&1 | FileCheck --check-prefix=O0 %s
; RUN: opt -mtriple=hexagon -S -passes='default<O2>' -print-pipeline-passes < %s 2>&1 | FileCheck %s

; CHECK: hexagon-loop-idiom
; CHECK: hexagon-vlcr
; O0: {{^}}function(ee-instrument<>),always-inline,coro-cond(coro-early,cgscc(coro-split),coro-cleanup,globaldce),alloc-token,function(annotation-remarks),verify,print{{$}}

define void @test_hexagon_passes() {
entry:
  ret void
}
