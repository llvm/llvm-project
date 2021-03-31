; RUN: opt < %s -S -passes="print<assumptions>,gvn-sink,loop-unroll" -unroll-count=3 | FileCheck %s
;
; This crashed because the cached assumption was replaced and the replacement
; was then in the cache twice.
;
; PR49043

@g = external global i32

define void @main() {
bb:
  %i1.i = load volatile i32, i32* @g
  %i32.i = icmp eq i32 %i1.i, 0
  call void @llvm.assume(i1 %i32.i) #3
  br label %bb4.i

bb4.i:                                            ; preds = %bb4.i, %bb
  %i.i = load volatile i32, i32* @g
  %i3.i = icmp eq i32 %i.i, 0
  call void @llvm.assume(i1 %i3.i) #3
  br label %bb4.i

func_1.exit:                                      ; No predecessors!
  unreachable
}

declare void @llvm.assume(i1)

; CHECK:  call void @llvm.assume(
; CHECK:  call void @llvm.assume(
; CHECK:  call void @llvm.assume(

