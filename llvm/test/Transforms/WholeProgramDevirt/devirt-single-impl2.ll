; Check that we can run WPD export using opt -passes=wholeprogramdevirt while
; loading/saving index from/to bitcode
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-as %p/Inputs/devirt-single-impl2-index.ll -o %t.index.bc
; RUN: opt %s -S -passes=wholeprogramdevirt -whole-program-visibility -wholeprogramdevirt-read-summary=%t.index.bc \
; RUN:     -wholeprogramdevirt-summary-action=export \
; RUN:     -wholeprogramdevirt-write-summary=%t2.index.bc -o /dev/null
; RUN: llvm-dis %t2.index.bc -o - | FileCheck %s

; Check that opt fails to use summaries which don't contain regular LTO module
; when performing export.
; RUN: llvm-as %p/Inputs/devirt-bad-index.ll -o %t-bad.index.bc
; RUN: not opt %s -S -passes=wholeprogramdevirt -whole-program-visibility -wholeprogramdevirt-read-summary=%t-bad.index.bc \
; RUN:     -wholeprogramdevirt-summary-action=export -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISSING-MODULE

; Check single impl devirtulation in summary
; CHECK: typeid: (name: "_ZTS1A", summary: (typeTestRes: (kind: unknown, sizeM1BitWidth: 0), wpdResolutions: ((offset: 0, wpdRes: (kind: singleImpl, singleImplName: "_ZNK1A1fEv"))))) ; guid

; MISSING-MODULE: combined summary should contain Regular LTO module

source_filename = "ld-temp.o"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { ptr }

$_ZTV1A = comdat any

@_ZTV1A = weak_odr hidden unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1A, ptr @_ZNK1A1fEv] }, comdat, align 8, !type !0, !type !1
@_ZTI1A = external hidden constant { ptr, ptr }, align 8
define available_externally hidden i32 @_ZNK1A1fEv(ptr %this) unnamed_addr align 2 {
entry:
  ret i32 3
}

!llvm.ident = !{!2}
!llvm.module.flags = !{!3, !4, !5}

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTSM1AKFivE.virtual"}
!2 = !{!"clang version 10.0.0 (trunk 373596)"}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!5 = !{i32 1, !"ThinLTO", i32 0}
