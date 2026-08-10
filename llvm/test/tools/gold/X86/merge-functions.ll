; RUN: llvm-as %s -o %t.bc
; RUN: llvm-as %p/Inputs/merge-functions-foo.ll -o %t-foo.bc
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 \
; RUN:    -plugin-opt=merge-functions \
; RUN:    -plugin-opt=save-temps \
; RUN:    -u main \
; RUN:    %t.bc %t-foo.bc \
; RUN:    -o %t-out
; RUN: llvm-dis %t-out.0.5.precodegen.bc -o - | FileCheck %s

; Check that we've merged foo and bar
; CHECK:      define dso_local noundef i32 @main()
; CHECK-NEXT:   tail call fastcc void @bar()
; CHECK-NEXT:   tail call fastcc void @bar()

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@_g = external local_unnamed_addr global i32, align 4

define dso_local i32 @bar(i32 noundef %0) #0 {
  %2 = add nsw i32 %0, 42
  store i32 %2, ptr @_g, align 4
  ret i32 %2
}

define dso_local noundef i32 @main() {
  %1 = tail call i32 @foo(i32 noundef 1)
  %2 = tail call i32 @bar(i32 noundef 1)
  ret i32 0
}

declare i32 @foo(i32 noundef) local_unnamed_addr #2

attributes #0 = { noinline }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"ThinLTO", i32 0}
!1 = !{i32 1, !"EnableSplitLTOUnit", i32 1}

^0 = module: (path: "merge-functions.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (name: "foo") ; guid = 6699318081062747564
^2 = gv: (name: "_g") ; guid = 9713702464056781075
^3 = gv: (name: "main", summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 1, live: 0, dsoLocal: 1, canAutoHide: 0), insts: 3, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 0, alwaysInline: 0, noUnwind: 1, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^1, tail: 1), (callee: ^4, tail: 1))))) ; guid = 15822663052811949562
^4 = gv: (name: "bar", summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 1, live: 0, dsoLocal: 1, canAutoHide: 0), insts: 3, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 1, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 1, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), refs: (^2)))) ; guid = 16434608426314478903
^5 = flags: 8
^6 = blockcount: 0
