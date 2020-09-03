; Simple checks of -print-changed functionality
;
; Note that (mostly) only the banners are checked.
;
; Simple functionality check.
; RUN: opt -S -print-changed -passes=instsimplify 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK0
;
; Check that only the passes that change the IR are printed and that the
; others (including g) are filtered out.
; RUN: opt -S -print-changed -passes=instsimplify -filter-print-funcs=f  2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK1
;
; Check that the reporting of IRs respects -print-module-scope
; RUN: opt -S -print-changed -passes=instsimplify -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK2
;
; Check that the reporting of IRs respects -print-module-scope
; RUN: opt -S -print-changed -passes=instsimplify -filter-print-funcs=f -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK3
;
; Check that reporting of multiple functions happens
; RUN: opt -S -print-changed -passes=instsimplify -filter-print-funcs="f,g" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK4
;
; Check that the reporting of IRs respects -filter-passes
; RUN: opt -S -print-changed -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK5
;
; Check that the reporting of IRs respects -filter-passes with multiple passes
; RUN: opt -S -print-changed -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass,InstSimplifyPass" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK6
;
; Check that the reporting of IRs respects both -filter-passes and -filter-print-funcs
; RUN: opt -S -print-changed -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass,InstSimplifyPass" -filter-print-funcs=f 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK7
;
; Check that the reporting of IRs respects -filter-passes, -filter-print-funcs and -print-module-scope
; RUN: opt -S -print-changed -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass,InstSimplifyPass" -filter-print-funcs=f -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK8
;
; Check that repeated passes that change the IR are printed and that the
; others (including g) are filtered out.  Note that the second time
; instsimplify is run on f, it does not change the IR
; RUN: opt -S -print-changed -passes="instsimplify,instsimplify" -filter-print-funcs=f  2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK9

define i32 @g() {
entry:
  %a = add i32 2, 3
  ret i32 %a
}

define i32 @f() {
entry:
  %a = add i32 2, 3
  ret i32 %a
}

; CHECK0: *** IR Dump At Start: ***
; CHECK0: ; ModuleID = '<stdin>'
; CHECK0: *** IR Dump After VerifierPass (module) omitted because no change ***
; CHECK0: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK0: *** IR Pass PassManager<llvm::Function> (function: g) ignored ***
; CHECK0: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK0: *** IR Pass PassManager<llvm::Function> (function: f) ignored ***
; CHECK0: *** IR Pass ModuleToFunctionPassAdaptor<llvm::PassManager<llvm::Function> > (module) ignored ***
; CHECK0: *** IR Dump After VerifierPass (module) omitted because no change ***
; CHECK0: *** IR Dump After PrintModulePass (module) omitted because no change ***

; CHECK1: *** IR Dump At Start: ***
; CHECK1: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK1: *** IR Dump After InstSimplifyPass *** (function: f)

; CHECK2: *** IR Dump At Start: ***
; CHECK2: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK2: ModuleID = '<stdin>'
; CHECK2: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK2: ModuleID = '<stdin>'

; CHECK3: *** IR Dump At Start: ***
; CHECK3: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK3: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK3: ModuleID = '<stdin>'

; CHECK4: *** IR Dump At Start: ***
; CHECK4: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK4: *** IR Dump After InstSimplifyPass *** (function: f)

; CHECK5: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK5: *** IR Dump At Start: *** (function: g)
; CHECK5: *** IR Dump After NoOpFunctionPass (function: g) omitted because no change ***
; CHECK5: *** IR Dump After InstSimplifyPass (function: f) filtered out ***
; CHECK5: *** IR Dump After NoOpFunctionPass (function: f) omitted because no change ***

; CHECK6: *** IR Dump At Start: *** (function: g)
; CHECK6: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK6: *** IR Dump After NoOpFunctionPass (function: g) omitted because no change ***
; CHECK6: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK6: *** IR Dump After NoOpFunctionPass (function: f) omitted because no change ***

; CHECK7: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK7: *** IR Dump After NoOpFunctionPass (function: g) filtered out ***
; CHECK7: *** IR Dump At Start: *** (function: f)
; CHECK7: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK7: *** IR Dump After NoOpFunctionPass (function: f) omitted because no change ***

; CHECK8: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK8: *** IR Dump After NoOpFunctionPass (function: g) filtered out ***
; CHECK8: *** IR Dump At Start: *** (function: f)
; CHECK8: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK8: ModuleID = '<stdin>'
; CHECK8: *** IR Dump After NoOpFunctionPass (function: f) omitted because no change ***

; CHECK9: *** IR Dump At Start: ***
; CHECK9: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK9: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK9: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK9: *** IR Dump After InstSimplifyPass (function: f) omitted because no change ***
