; This test checks if two similar functions, f1 and f2, can be merged locally within a single module
; while parameterizing a difference in their global variables, g1 and g2.
; To achieve this, we create two instances of the global merging function, f1.Tgm and f2.Tgm,
; which are tail-called from thunks g1 and g2 respectively.
; These identical functions, f1.Tgm and f2.Tgm, will be folded by the linker via Identical Code Folding (IFC).

; RUN: opt -module-summary -module-hash %s -o %t

; RUN: llvm-lto2 run -enable-global-merge-func=false %t -o %tout-nomerge \
; RUN:    -r %t,_f1,px \
; RUN:    -r %t,_f2,px \
; RUN:    -r %t,_g,l -r %t,_g1,l -r %t,_g2,l
; RUN: llvm-nm %tout-nomerge.1 | FileCheck %s --check-prefix=NOMERGE
; RUN: llvm-lto2 run -enable-global-merge-func=true %t -o %tout-merge \
; RUN:    -r %t,_f1,px \
; RUN:    -r %t,_f2,px \
; RUN:    -r %t,_g,l -r %t,_g1,l -r %t,_g2,l
; RUN: llvm-nm %tout-merge.1 | FileCheck %s --check-prefix=GLOBALMERGE
; RUN: llvm-objdump -d %tout-merge.1 | FileCheck %s --check-prefix=THUNK

; NOMERGE-NOT: _f1.Tgm
; GLOBALMERGE: _f1.Tgm
; GLOBALMERGE: _f2.Tgm

; THUNK: <_f1>:
; THUNK-NEXT: adrp x1,
; THUNK-NEXT: ldr x1, [x1]
; THUNK-NEXT: b

; THUNK: <_f2>:
; THUNK-NEXT: adrp x1,
; THUNK-NEXT: ldr x1, [x1]
; THUNK-NEXT: b

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-unknown-ios12.0.0"

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
