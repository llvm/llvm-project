; RUN: opt -passes='print<ir2vec>' -o /dev/null -ir2vec-vocab-path=%S/Inputs/dummy_3D_vocab.json %s 2>&1 | FileCheck %s

define void @bar2(ptr %foo)  {
  store i32 0, ptr %foo, align 4
  tail call void @llvm.dbg.value(metadata !{}, i64 0, metadata !{}, metadata !{})
  ret void
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

; CHECK: Instruction vectors:
; CHECK-NEXT: Instruction:   store i32 0, ptr %foo, align 4 [ 7.00  8.00  9.00 ]
; CHECK-NEXT: Instruction:   ret void [ 0.00  0.00  0.00 ]
