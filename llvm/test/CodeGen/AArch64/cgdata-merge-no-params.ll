; This test verifies whether two identical functions, f1 and f2, can be merged
; locally using the global merge function.
; The functions, f1.Tgm and f2.Tgm, will be folded by the linker through
; Identical Code Folding (ICF).
; While identical functions can already be folded by the linker, creating this
; canonical form can be beneficial in downstream passes. This merging process
; can be controlled by the -global-merging-skip-no-params option.

; RUN: llc -mtriple=arm64-apple-darwin -enable-global-merge-func=true -global-merging-skip-no-params=false < %s | FileCheck %s --check-prefix=MERGE
; RUN: llc -mtriple=arm64-apple-darwin -enable-global-merge-func=true -global-merging-skip-no-params=true < %s | FileCheck %s --implicit-check-not=".Tgm"

; MERGE: _f1.Tgm
; MERGE: _f2.Tgm

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
  %1 = load volatile i32, i32* @g1, align 4
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, 1
  ret i32 %add
}
