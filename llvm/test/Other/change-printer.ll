; Simple checks of -print-changed functionality
;
; Note that (mostly) only the banners are checked.
;
; Simple functionality check.
; RUN: opt -S -print-changed -passes=instsimplify 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-SIMPLE
;
; Check that only the passes that change the IR are printed and that the
; others (including g) are filtered out.
; RUN: opt -S -print-changed -passes=instsimplify -filter-print-funcs=f  2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FUNC-FILTER
;
; Check that the reporting of IRs respects -print-module-scope
; RUN: opt -S -print-changed -passes=instsimplify -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-PRINT-MOD-SCOPE
;
; Check that the reporting of IRs respects -print-module-scope
; RUN: opt -S -print-changed -passes=instsimplify -filter-print-funcs=f -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FUNC-FILTER-MOD-SCOPE
;
; Check that reporting of multiple functions happens
; RUN: opt -S -print-changed -passes=instsimplify -filter-print-funcs="f,g" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FILTER-MULT-FUNC
;
; Check that the reporting of IRs respects -filter-passes
; RUN: opt -S -print-changed -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FILTER-PASSES
;
; Check that the reporting of IRs respects -filter-passes with multiple passes
; RUN: opt -S -print-changed -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass,InstSimplifyPass" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FILTER-MULT-PASSES
;
; Check that the reporting of IRs respects both -filter-passes and -filter-print-funcs
; RUN: opt -S -print-changed -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass,InstSimplifyPass" -filter-print-funcs=f 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FILTER-FUNC-PASSES
;
; Check that the reporting of IRs respects -filter-passes, -filter-print-funcs and -print-module-scope
; RUN: opt -S -print-changed -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass,InstSimplifyPass" -filter-print-funcs=f -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FILTER-FUNC-PASSES-MOD-SCOPE
;
; Check that repeated passes that change the IR are printed and that the
; others (including g) are filtered out.  Note that the second time
; instsimplify is run on f, it does not change the IR
; RUN: opt -S -print-changed -passes="instsimplify,instsimplify" -filter-print-funcs=f  2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-MULT-PASSES-FILTER-FUNC
;
; Simple print-before-changed functionality check.
; RUN: opt -S -print-changed -print-before-changed -passes=instsimplify 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-SIMPLE-BEFORE
;
; Check print-before-changed obeys the function filtering
; RUN: opt -S -print-changed -print-before-changed -passes=instsimplify -filter-print-funcs=f  2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FUNC-FILTER-BEFORE
;
; Check that the reporting of IRs with -print-before-changed respects -print-module-scope
; RUN: opt -S -print-changed -print-before-changed -passes=instsimplify -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-PRINT-MOD-SCOPE-BEFORE
;

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

; CHECK-SIMPLE: *** IR Dump At Start: ***
; CHECK-SIMPLE-NEXT: ; ModuleID = {{.+}}
; CHECK-SIMPLE: *** IR Dump After VerifierPass (module) omitted because no change ***
; CHECK-SIMPLE: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK-SIMPLE-NEXT: define i32 @g()
; CHECK-SIMPLE: *** IR Pass PassManager{{.*}} (function: g) ignored ***
; CHECK-SIMPLE: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-SIMPLE-NEXT: define i32 @f()
; CHECK-SIMPLE: *** IR Pass PassManager{{.*}} (function: f) ignored ***
; CHECK-SIMPLE: *** IR Pass ModuleToFunctionPassAdaptor (module) ignored ***
; CHECK-SIMPLE: *** IR Dump After VerifierPass (module) omitted because no change ***
; CHECK-SIMPLE: *** IR Dump After PrintModulePass (module) omitted because no change ***
; CHECK-SIMPLE-NOT: *** IR

; CHECK-FUNC-FILTER: *** IR Dump At Start: ***
; CHECK-FUNC-FILTER-NEXT: ; ModuleID = {{.+}}
; CHECK-FUNC-FILTER: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK-FUNC-FILTER: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-FUNC-FILTER-NEXT: define i32 @f()

; CHECK-PRINT-MOD-SCOPE: *** IR Dump At Start: ***
; CHECK-PRINT-MOD-SCOPE-NEXT: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK-PRINT-MOD-SCOPE-NEXT: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-PRINT-MOD-SCOPE-NEXT: ModuleID = {{.+}}

; CHECK-FUNC-FILTER-MOD-SCOPE: *** IR Dump At Start: ***
; CHECK-FUNC-FILTER-MOD-SCOPE-NEXT: ; ModuleID = {{.+}}
; CHECK-FUNC-FILTER-MOD-SCOPE: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK-FUNC-FILTER-MOD-SCOPE: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-FUNC-FILTER-MOD-SCOPE-NEXT: ModuleID = {{.+}}

; CHECK-FILTER-MULT-FUNC: *** IR Dump At Start: ***
; CHECK-FILTER-MULT-FUNC-NEXT: ; ModuleID = {{.+}}
; CHECK-FILTER-MULT-FUNC: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK-FILTER-MULT-FUNC-NEXT: define i32 @g()
; CHECK-FILTER-MULT-FUNC: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-FILTER-MULT-FUNC-NEXT: define i32 @f()

; CHECK-FILTER-PASSES: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK-FILTER-PASSES: *** IR Dump At Start: *** (function: g)
; CHECK-FILTER-PASSES-NEXT: ; ModuleID = {{.+}}
; CHECK-FILTER-PASSES: *** IR Dump After NoOpFunctionPass (function: g) omitted because no change ***
; CHECK-FILTER-PASSES: *** IR Dump After InstSimplifyPass (function: f) filtered out ***
; CHECK-FILTER-PASSES: *** IR Dump After NoOpFunctionPass (function: f) omitted because no change ***

; CHECK-FILTER-MULT-PASSES: *** IR Dump At Start: *** (function: g)
; CHECK-FILTER-MULT-PASSES-NEXT: ; ModuleID = {{.+}}
; CHECK-FILTER-MULT-PASSES: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK-FILTER-MULT-PASSES-NEXT: define i32 @g()
; CHECK-FILTER-MULT-PASSES: *** IR Dump After NoOpFunctionPass (function: g) omitted because no change ***
; CHECK-FILTER-MULT-PASSES: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-FILTER-MULT-PASSES-NEXT: define i32 @f()
; CHECK-FILTER-MULT-PASSES: *** IR Dump After NoOpFunctionPass (function: f) omitted because no change ***

; CHECK-FILTER-FUNC-PASSES: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK-FILTER-FUNC-PASSES: *** IR Dump After NoOpFunctionPass (function: g) filtered out ***
; CHECK-FILTER-FUNC-PASSES: *** IR Dump At Start: *** (function: f)
; CHECK-FILTER-FUNC-PASSES-NEXT: ; ModuleID = {{.+}}
; CHECK-FILTER-FUNC-PASSES: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-FILTER-FUNC-PASSES-NEXT: define i32 @f()
; CHECK-FILTER-FUNC-PASSES: *** IR Dump After NoOpFunctionPass (function: f) omitted because no change ***

; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE: *** IR Dump After NoOpFunctionPass (function: g) filtered out ***
; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE: *** IR Dump At Start: *** (function: f)
; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE-NEXT: ; ModuleID = {{.+}}
; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE-NEXT: ModuleID = {{.+}}
; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE: *** IR Dump After NoOpFunctionPass (function: f) omitted because no change ***

; CHECK-MULT-PASSES-FILTER-FUNC: *** IR Dump At Start: ***
; CHECK-MULT-PASSES-FILTER-FUNC-NEXT: ; ModuleID = {{.+}}
; CHECK-MULT-PASSES-FILTER-FUNC: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK-MULT-PASSES-FILTER-FUNC: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK-MULT-PASSES-FILTER-FUNC: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-MULT-PASSES-FILTER-FUNC-NEXT: define i32 @f()
; CHECK-MULT-PASSES-FILTER-FUNC: *** IR Dump After InstSimplifyPass (function: f) omitted because no change ***

; CHECK-SIMPLE-BEFORE: *** IR Dump At Start: ***
; CHECK-SIMPLE-BEFORE-NEXT: ; ModuleID = {{.+}}
; CHECK-SIMPLE-BEFORE: *** IR Dump After VerifierPass (module) omitted because no change ***
; CHECK-SIMPLE-BEFORE: *** IR Dump Before InstSimplifyPass *** (function: g)
; CHECK-SIMPLE-BEFORE-NEXT: define i32 @g()
; CHECK-SIMPLE-BEFORE: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK-SIMPLE-BEFORE-NEXT: define i32 @g()
; CHECK-SIMPLE-BEFORE: *** IR Pass PassManager{{.*}} (function: g) ignored ***
; CHECK-SIMPLE-BEFORE: *** IR Dump Before InstSimplifyPass *** (function: f)
; CHECK-SIMPLE-BEFORE-NEXT: define i32 @f()
; CHECK-SIMPLE-BEFORE: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-SIMPLE-BEFORE-NEXT: define i32 @f()
; CHECK-SIMPLE-BEFORE: *** IR Pass PassManager{{.*}} (function: f) ignored ***
; CHECK-SIMPLE-BEFORE: *** IR Pass ModuleToFunctionPassAdaptor (module) ignored ***
; CHECK-SIMPLE-BEFORE: *** IR Dump After VerifierPass (module) omitted because no change ***
; CHECK-SIMPLE-BEFORE: *** IR Dump After PrintModulePass (module) omitted because no change ***
; CHECK-SIMPLE-BEFORE-NOT: *** IR

; CHECK-FUNC-FILTER-BEFORE: *** IR Dump At Start: ***
; CHECK-FUNC-FILTER-BEFORE-NEXT: ; ModuleID = {{.+}}
; CHECK-FUNC-FILTER-BEFORE: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK-FUNC-FILTER-BEFORE: *** IR Dump Before InstSimplifyPass *** (function: f)
; CHECK-FUNC-FILTER-BEFORE-NEXT: define i32 @f()
; CHECK-FUNC-FILTER-BEFORE: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-FUNC-FILTER-BEFORE-NEXT: define i32 @f()

; CHECK-PRINT-MOD-SCOPE-BEFORE: *** IR Dump At Start: ***
; CHECK-PRINT-MOD-SCOPE-BEFORE-NEXT: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE-BEFORE: *** IR Dump Before InstSimplifyPass *** (function: g)
; CHECK-PRINT-MOD-SCOPE-BEFORE-NEXT: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE-BEFORE: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK-PRINT-MOD-SCOPE-BEFORE-NEXT: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE-BEFORE: *** IR Dump Before InstSimplifyPass *** (function: f)
; CHECK-PRINT-MOD-SCOPE-BEFORE-NEXT: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE-BEFORE: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-PRINT-MOD-SCOPE-BEFORE-NEXT: ModuleID = {{.+}}
