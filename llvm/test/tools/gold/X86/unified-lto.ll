; Check that we can use full LTO with gold plugin when inputs
; are compiled using unified LTO pipeline
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-as %p/Inputs/unified-lto-foo.ll -o %t-foo.bc
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 \
; RUN:    -plugin-opt=unifiedlto \
; RUN:    -plugin-opt=save-temps \
; RUN:    -u main \
; RUN:    %t.bc %t-foo.bc \
; RUN:    -o %t-out
; RUN: llvm-dis %t-out.0.5.precodegen.bc -o - | FileCheck %s

; Check thin LTO as well
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 \
; RUN:    -plugin-opt=unifiedlto \
; RUN:    -plugin-opt=thinlto \
; RUN:    -plugin-opt=save-temps \
; RUN:    -u main \
; RUN:    %t.bc %t-foo.bc \
; RUN:    -o %t-out
; RUN: llvm-dis %t.bc.5.precodegen.bc -o - | FileCheck %s --check-prefix=THIN

; Constant propagation is not supported by thin LTO.
; With full LTO we fold argument into constant 43
; CHECK:       define dso_local noundef i32 @main()
; CHECK-NEXT:    tail call fastcc void @foo()
; CHECK-NEXT:    ret i32 43

; CHECK:       define internal fastcc void @foo()
; CHECK-NEXT:    store i32 43, ptr @_g, align 4

; ThinLTO doesn't import foo, because the latter has noinline attribute
; THIN:      define dso_local i32 @main()
; THIN-NEXT:   %1 = tail call i32 @foo(i32 noundef 1)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define dso_local i32 @main() {
  %1 = tail call i32 @foo(i32 noundef 1)
  ret i32 %1
}

declare i32 @foo(i32 noundef)

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!1 = !{i32 1, !"UnifiedLTO", i32 1}

^0 = module: (path: "unified-lto.o", hash: (2850108895, 1189778381, 479678006, 1191715608, 4095117687))
^1 = gv: (name: "foo") ; guid = 6699318081062747564
^2 = gv: (name: "main", summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0), insts: 2, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 0, alwaysInline: 0, noUnwind: 1, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^1, tail: 1))))) ; guid = 15822663052811949562
^3 = flags: 520
^4 = blockcount: 0
