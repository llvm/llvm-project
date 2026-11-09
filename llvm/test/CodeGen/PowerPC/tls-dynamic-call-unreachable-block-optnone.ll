; REQUIRES: asserts
; RUN: not --crash llc -mtriple=powerpc64le-unknown-linux-gnu -relocation-model=pic -enable-pipeliner -filetype=null %s 2>&1 | FileCheck %s

; CHECK: does not have a corresponding definition on every path

; FIXME: Merge with tls-dynamic-call-unreachable-block.ll when fixed

; FastISel leaves an unreachable block using a value defined in the
; entry block. LiveIntervals doesn't handle code with unreachable
; blocks, and it's forced to be computed by enabling the pipeliner.

declare void @g(i64)

define void @unreachable_block_use_optnone(i64 %a) nounwind optnone noinline {
entry:
  br i1 false, label %dead, label %exit

dead:
  call void @g(i64 %a)
  br label %exit

exit:
  ret void
}
