; RUN: llvm-as %s -o %t.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    -shared %t.o -o %t2
; RUN: llvm-nm %t2 | FileCheck %s
; CHECK: T main

@main.L = internal unnamed_addr constant [3 x i8*] [i8* blockaddress(@main, %L1), i8* blockaddress(@main, %L2), i8* null], align 16

define i32 @main() #0 {
entry:
  br label %L1

L1:                                               ; preds = %entry, %L1
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %L1 ]
  %inc = add i32 %i.0, 1
  %idxprom = zext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds [3 x i8*], [3 x i8*]* @main.L, i64 0, i64 %idxprom
  %0 = load i8*, i8** %arrayidx, align 8, !tbaa !1
  indirectbr i8* %0, [label %L1, label %L2]

L2:                                               ; preds = %L1
  ret i32 0
}

!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
