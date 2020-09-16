; Simple checks of -print-changed functionality
;
; Note that (mostly) only the banners are checked.
;
; Simple functionality check.
; RUN: opt -S -print-changed -passes=instsimplify 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK_SIMPLE
;
; Check that only the passes that change the IR are printed and that the
; others (including g) are filtered out.
; RUN: opt -S -print-changed -passes=instsimplify -filter-print-funcs=f  2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK_FUNC_FILTER
;
; Check that the reporting of IRs respects -print-module-scope
; RUN: opt -S -print-changed -passes=instsimplify -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK_PRINT_MOD_SCOPE
;
; Check that the reporting of IRs respects -print-module-scope
; RUN: opt -S -print-changed -passes=instsimplify -filter-print-funcs=f -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK_FUNC_FILTER_MOD_SCOPE
;
; Check that reporting of multiple functions happens
; RUN: opt -S -print-changed -passes=instsimplify -filter-print-funcs="f,g" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK_FILTER_MULT_FUNC
;
; Check that the reporting of IRs respects -filter-passes
; RUN: opt -S -print-changed -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK_FILTER_PASSES
;
; Check that the reporting of IRs respects -filter-passes with multiple passes
; RUN: opt -S -print-changed -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass,InstSimplifyPass" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK_FILTER_MULT_PASSES
;
; Check that the reporting of IRs respects both -filter-passes and -filter-print-funcs
; RUN: opt -S -print-changed -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass,InstSimplifyPass" -filter-print-funcs=f 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK_FILTER_FUNC_PASSES
;
; Check that the reporting of IRs respects -filter-passes, -filter-print-funcs and -print-module-scope
; RUN: opt -S -print-changed -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass,InstSimplifyPass" -filter-print-funcs=f -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK_FILTER_FUNC_PASSES_MOD_SCOPE
;
; Check that repeated passes that change the IR are printed and that the
; others (including g) are filtered out.  Note that the second time
; instsimplify is run on f, it does not change the IR
; RUN: opt -S -print-changed -passes="instsimplify,instsimplify" -filter-print-funcs=f  2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK_MULT_PASSES_FILTER_FUNC

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

; CHECK_SIMPLE: *** IR Dump At Start: ***
; CHECK_SIMPLE: ; ModuleID = '<stdin>'
; CHECK_SIMPLE: *** IR Dump After VerifierPass (module) omitted because no change ***
; CHECK_SIMPLE: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK_SIMPLE: *** IR Pass PassManager<llvm::Function> (function: g) ignored ***
; CHECK_SIMPLE: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK_SIMPLE: *** IR Pass PassManager<llvm::Function> (function: f) ignored ***
; CHECK_SIMPLE: *** IR Pass ModuleToFunctionPassAdaptor<llvm::PassManager<llvm::Function>{{ ?}}> (module) ignored ***
; CHECK_SIMPLE: *** IR Dump After VerifierPass (module) omitted because no change ***
; CHECK_SIMPLE: *** IR Dump After PrintModulePass (module) omitted because no change ***

; CHECK_FUNC_FILTER: *** IR Dump At Start: ***
; CHECK_FUNC_FILTER: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK_FUNC_FILTER: *** IR Dump After InstSimplifyPass *** (function: f)

; CHECK_PRINT_MOD_SCOPE: *** IR Dump At Start: ***
; CHECK_PRINT_MOD_SCOPE: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK_PRINT_MOD_SCOPE: ModuleID = '<stdin>'
; CHECK_PRINT_MOD_SCOPE: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK_PRINT_MOD_SCOPE: ModuleID = '<stdin>'

; CHECK_FUNC_FILTER_MOD_SCOPE: *** IR Dump At Start: ***
; CHECK_FUNC_FILTER_MOD_SCOPE: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK_FUNC_FILTER_MOD_SCOPE: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK_FUNC_FILTER_MOD_SCOPE: ModuleID = '<stdin>'

; CHECK_FILTER_MULT_FUNC: *** IR Dump At Start: ***
; CHECK_FILTER_MULT_FUNC: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK_FILTER_MULT_FUNC: *** IR Dump After InstSimplifyPass *** (function: f)

; CHECK_FILTER_PASSES: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK_FILTER_PASSES: *** IR Dump At Start: *** (function: g)
; CHECK_FILTER_PASSES: *** IR Dump After NoOpFunctionPass (function: g) omitted because no change ***
; CHECK_FILTER_PASSES: *** IR Dump After InstSimplifyPass (function: f) filtered out ***
; CHECK_FILTER_PASSES: *** IR Dump After NoOpFunctionPass (function: f) omitted because no change ***

; CHECK_FILTER_MULT_PASSES: *** IR Dump At Start: *** (function: g)
; CHECK_FILTER_MULT_PASSES: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK_FILTER_MULT_PASSES: *** IR Dump After NoOpFunctionPass (function: g) omitted because no change ***
; CHECK_FILTER_MULT_PASSES: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK_FILTER_MULT_PASSES: *** IR Dump After NoOpFunctionPass (function: f) omitted because no change ***

; CHECK_FILTER_FUNC_PASSES: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK_FILTER_FUNC_PASSES: *** IR Dump After NoOpFunctionPass (function: g) filtered out ***
; CHECK_FILTER_FUNC_PASSES: *** IR Dump At Start: *** (function: f)
; CHECK_FILTER_FUNC_PASSES: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK_FILTER_FUNC_PASSES: *** IR Dump After NoOpFunctionPass (function: f) omitted because no change ***

; CHECK_FILTER_FUNC_PASSES_MOD_SCOPE: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK_FILTER_FUNC_PASSES_MOD_SCOPE: *** IR Dump After NoOpFunctionPass (function: g) filtered out ***
; CHECK_FILTER_FUNC_PASSES_MOD_SCOPE: *** IR Dump At Start: *** (function: f)
; CHECK_FILTER_FUNC_PASSES_MOD_SCOPE: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK_FILTER_FUNC_PASSES_MOD_SCOPE: ModuleID = '<stdin>'
; CHECK_FILTER_FUNC_PASSES_MOD_SCOPE: *** IR Dump After NoOpFunctionPass (function: f) omitted because no change ***

; CHECK_MULT_PASSES_FILTER_FUNC: *** IR Dump At Start: ***
; CHECK_MULT_PASSES_FILTER_FUNC: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK_MULT_PASSES_FILTER_FUNC: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK_MULT_PASSES_FILTER_FUNC: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK_MULT_PASSES_FILTER_FUNC: *** IR Dump After InstSimplifyPass (function: f) omitted because no change ***
