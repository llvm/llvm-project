; The IR below was crafted so as:
; 1) To have a loop, so we create a loop pass manager
; 2) To be "immutable" in the sense that no pass in the standard
;    pipeline will modify it.
; Since no transformations take place, we don't expect any analyses
; to be invalidated.
; Any invalidation that shows up here is a bug, unless we started modifying
; the IR, in which case we need to make it immutable harder.

; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes='default<O0>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK,CHECK-DEFAULT
; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes='thinlto-pre-link<O0>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK,CHECK-DEFAULT,CHECK-PRE-LINK
; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes='lto-pre-link<O0>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK,CHECK-DEFAULT,CHECK-PRE-LINK
; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes='thinlto<O0>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK,CHECK-THINLTO
; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes='lto<O0>' -S %s 2>&1 \
; RUN:     | FileCheck %s --check-prefixes=CHECK,CHECK-LTO

; CHECK: Starting llvm::Module pass manager run.
; CHECK-DEFAULT-NEXT: Running pass: AlwaysInlinerPass
; CHECK-DEFAULT-NEXT: Running analysis: InnerAnalysisManagerProxy
; CHECK-DEFAULT-NEXT: Running analysis: ProfileSummaryAnalysis
; CHECK-PRE-LINK-NEXT: Running pass: CanonicalizeAliasesPass
; CHECK-PRE-LINK-NEXT: Running pass: NameAnonGlobalPass
; CHECK-THINLTO-NEXT: Running pass: Annotation2MetadataPass
; CHECK-LTO-NEXT: Running pass: Annotation2MetadataPass
; CHECK-LTO-NEXT: Running pass: WholeProgramDevirtPass
; CHECK-LTO-NEXT: Running analysis: InnerAnalysisManagerProxy
; CHECK-LTO-NEXT: Running pass: LowerTypeTestsPass
; CHECK-LTO-NEXT: Running pass: LowerTypeTestsPass
; CHECK-LTO-NEXT: Running pass: AnnotationRemarksPass
; CHECK-NEXT: Running pass: PrintModulePass

; Make sure we get the IR back out without changes when we print the module.
; CHECK-LABEL: define void @foo(i32 %n) local_unnamed_addr {
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %loop
; CHECK:      loop:
; CHECK-NEXT:   %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
; CHECK-NEXT:   %iv.next = add i32 %iv, 1
; CHECK-NEXT:   tail call void @bar()
; CHECK-NEXT:   %cmp = icmp eq i32 %iv, %n
; CHECK-NEXT:   br i1 %cmp, label %exit, label %loop
; CHECK:      exit:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
;
; CHECK-NEXT: Finished llvm::Module pass manager run.

declare void @bar() local_unnamed_addr

define void @foo(i32 %n) local_unnamed_addr {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add i32 %iv, 1
  tail call void @bar()
  %cmp = icmp eq i32 %iv, %n
  br i1 %cmp, label %exit, label %loop
exit:
  ret void
}
