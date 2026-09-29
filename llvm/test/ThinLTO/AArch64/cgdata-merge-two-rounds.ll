; TODO: This test checks if the how similar functions are handled during global outlining
; by repeating the codegen via -codegen-data-thinlto-two-rounds=true.

; RUN: rm -rf %t; split-file %s %t

; RUN: opt -module-summary -module-hash %t/foo.ll -o %t-foo.bc
; RUN: opt -module-summary -module-hash %t/goo.ll -o %t-goo.bc

; RUN: llvm-lto2 run -enable-global-merge-func=true -codegen-data-thinlto-two-rounds=true %t-foo.bc %t-goo.bc -o %tout \
; RUN:    -r %t-foo.bc,_f1,px \
; RUN:    -r %t-goo.bc,_f2,px \
; RUN:    -r %t-foo.bc,_g,l -r %t-foo.bc,_g1,l -r %t-foo.bc,_g2,l \
; RUN:    -r %t-goo.bc,_g,l -r %t-goo.bc,_g1,l -r %t-goo.bc,_g2,l
; RUN: llvm-nm %tout.1 | FileCheck %s --check-prefix=OUT1
; RUN: llvm-nm %tout.2 | FileCheck %s --check-prefix=OUT2
; RUN: llvm-objdump -d %tout.1 | FileCheck %s --check-prefix=THUNK1
; RUN: llvm-objdump -d %tout.2 | FileCheck %s --check-prefix=THUNK2

; OUT1: _f1.Tgm
; OUT2: _f2.Tgm

; THUNK1: <_f1>:
; THUNK1-NEXT: adrp x1,
; THUNK1-NEXT: ldr x1, [x1]
; THUNK1-NEXT: b

; THUNK2: <_f2>:
; THUNK2-NEXT: adrp x1,
; THUNK2-NEXT: ldr x1, [x1]
; THUNK2-NEXT: b

;--- foo.ll
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

;--- goo.ll
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-unknown-ios12.0.0"

@g = external local_unnamed_addr global [0 x i32], align 4
@g1 = external global i32, align 4
@g2 = external global i32, align 4

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
