; Check interaction between LTO and LLVM mangling escape char, see #57864.

; RUN: split-file %s %t
; RUN: opt %t/hide-me.ll -o %t/hide-me.bc
; RUN: opt %t/ref.ll -o %t/ref.bc
; RUN: llvm-lto2 run \
; RUN:               -r %t/hide-me.bc,_hide_me,p \
; RUN:               -r %t/ref.bc,_main,plx  \
; RUN:               -r %t/ref.bc,_hide_me,l \
; RUN:               --select-save-temps=precodegen \
; RUN:               -o %t/out \
; RUN:               %t/hide-me.bc  %t/ref.bc
; RUN: llvm-dis %t/out.0.5.precodegen.bc -o - | FileCheck %s


;--- hide-me.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.0"

@"\01_hide_me" = hidden local_unnamed_addr global i8 8, align 1

;--- ref.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.0"

@hide_me = external local_unnamed_addr global i8

define i8 @main() {
  %1 = load i8, ptr @hide_me, align 1
  ret i8 %1
}


; CHECK: @"\01_hide_me" = hidden local_unnamed_addr global i8 8, align 1
; CHECK: @hide_me = external dso_local local_unnamed_addr global i8

; CHECK: define dso_local i8 @main() local_unnamed_addr #0 {
; CHECK:   %1 = load i8, ptr @hide_me, align 1
; CHECK:   ret i8 %1
; CHECK: }

