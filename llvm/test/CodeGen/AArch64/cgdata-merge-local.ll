; This test checks if two similar functions, f1 and f2, can be merged locally within a single module
; while parameterizing a difference in their global variables, g1 and g2.
; To achieve this, we create two instances of the global merging function, f1.Tgm and f2.Tgm,
; which are tail-called from thunks f1 and f2 respectively.
; These identical functions, f1.Tgm and f2.Tgm, will be folded by the linker via Identical Code Folding (ICF).

; RUN: opt -mtriple=arm64-apple-darwin -S --passes=global-merge-func %s | FileCheck %s

; A merging instance is created with additional parameter.
; CHECK: define internal i32 @f1.Tgm(i32 %0, ptr %1)
; CHECK-NEXT: entry:
; CHECK-NEXT:  %idxprom = sext i32 %0 to i64
; CHECK-NEXT:  %arrayidx = getelementptr inbounds [0 x i32], ptr @g, i64 0, i64 %idxprom
; CHECK-NEXT:  %2 = load i32, ptr %arrayidx, align 4
; CHECK-NEXT:  %3 = load volatile i32, ptr %1, align 4
; CHECK-NEXT:  %mul = mul nsw i32 %3, %2
; CHECK-NEXT:  %add = add nsw i32 %mul, 1
; CHECK-NEXT:  ret i32 %add

; The original function becomes a thunk passing g1.
; CHECK: define i32 @f1(i32 %a)
; CHECK-NEXT:  %1 = tail call i32 @f1.Tgm(i32 %a, ptr @g1)
; CHECK-NEXT:  ret i32 %1

; A same sequence is produced for f2.Tgm.
; CHECK: define internal i32 @f2.Tgm(i32 %0, ptr %1)
; CHECK-NEXT: entry:
; CHECK-NEXT:  %idxprom = sext i32 %0 to i64
; CHECK-NEXT:  %arrayidx = getelementptr inbounds [0 x i32], ptr @g, i64 0, i64 %idxprom
; CHECK-NEXT:  %2 = load i32, ptr %arrayidx, align 4
; CHECK-NEXT:  %3 = load volatile i32, ptr %1, align 4
; CHECK-NEXT:  %mul = mul nsw i32 %3, %2
; CHECK-NEXT:  %add = add nsw i32 %mul, 1
; CHECK-NEXT:  ret i32 %add

; The original function becomes a thunk passing g2.
; CHECK: define i32 @f2(i32 %a)
; CHECK-NEXT:  %1 = tail call i32 @f2.Tgm(i32 %a, ptr @g2)
; CHECK-NEXT:  ret i32 %1

; RUN: llc -mtriple=arm64-apple-darwin -enable-global-merge-func=true < %s | FileCheck %s --check-prefix=MERGE
; RUN: llc -mtriple=arm64-apple-darwin -enable-global-merge-func=false < %s | FileCheck %s --check-prefix=NOMERGE

; MERGE: _f1.Tgm
; MERGE: _f2.Tgm

; NOMERGE-NOT: _f1.Tgm
; NOMERGE-NOT: _f2.Tgm

@g = external local_unnamed_addr global [0 x i32], align 4
@g1 = external global i32, align 4
@g2 = external global i32, align 4

define i32 @f1(i32 %a) {
entry:
  %idxprom = sext i32 %a to i64
  %arrayidx = getelementptr inbounds [0 x i32], [0 x i32]* @g, i64 0, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %1 = load volatile i32, i32* @g1, align 4
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, 1
  ret i32 %add
}

define i32 @f2(i32 %a) {
entry:
  %idxprom = sext i32 %a to i64
  %arrayidx = getelementptr inbounds [0 x i32], [0 x i32]* @g, i64 0, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %1 = load volatile i32, i32* @g2, align 4
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, 1
  ret i32 %add
}
