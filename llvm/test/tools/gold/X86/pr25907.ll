; RUN: llvm-as %s -o %t.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 \
; RUN:    -shared %t.o -o %t2
; RUN: llvm-nm %t2 | FileCheck %s
; CHECK: T main

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@main.L = internal unnamed_addr constant [3 x ptr] [ptr blockaddress(@main, %L1), ptr blockaddress(@main, %L2), ptr null], align 16

define i32 @main() #0 {
entry:
  br label %L1

L1:                                               ; preds = %entry, %L1
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %L1 ]
  %inc = add i32 %i.0, 1
  %idxprom = zext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds [3 x ptr], ptr @main.L, i64 0, i64 %idxprom
  %0 = load ptr, ptr %arrayidx, align 8, !tbaa !1
  indirectbr ptr %0, [label %L1, label %L2]

L2:                                               ; preds = %L1
  ret i32 0
}

!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
