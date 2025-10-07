; RUN: llvm-link %S/Inputs/errno-tbaa-metadata.ll %S/Inputs/errno-tbaa-cxx-metadata.ll -S -o - | FileCheck %s --check-prefix=CHECK-MERGE
; RUN: llvm-link %S/Inputs/errno-tbaa-metadata.ll %S/Inputs/errno-tbaa-metadata.ll -S -o - | FileCheck %s --check-prefix=CHECK-DEDUP

; Ensure merging when linking modules w/ different errno TBAA hierarchies.
; CHECK-MERGE: !llvm.errno.tbaa = !{![[NODE0:[0-9]+]], ![[NODE1:[0-9]+]]}

; Ensure deduplication when linking modules w/ identical errno TBAA nodes.
; CHECK-DEDUP: !llvm.errno.tbaa = !{![[NODE:[0-9]+]]}
