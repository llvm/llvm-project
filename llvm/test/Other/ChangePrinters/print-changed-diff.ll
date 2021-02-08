; Simple checks of -print-changed=diff
;
; Note that (mostly) only the banners are checked.
;
; Simple functionality check.
; RUN: opt -S -print-changed=diff -passes=instsimplify 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-DIFF-SIMPLE
;
; Check that only the passes that change the IR are printed and that the
; others (including g) are filtered out.
; RUN: opt -S -print-changed=diff -passes=instsimplify -filter-print-funcs=f  2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-DIFF-FUNC-FILTER
;
; Check that the reporting of IRs respects is not affected by
; -print-module-scope
; RUN: opt -S -print-changed=diff -passes=instsimplify -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-DIFF-PRINT-MOD-SCOPE
;
; Check that reporting of multiple functions happens
; RUN: opt -S -print-changed=diff -passes=instsimplify -filter-print-funcs="f,g" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-DIFF-FILTER-MULT-FUNC
;
; Check that the reporting of IRs respects -filter-passes
; RUN: opt -S -print-changed=diff -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-DIFF-FILTER-PASSES
;
; Check that the reporting of IRs respects -filter-passes with multiple passes
; RUN: opt -S -print-changed=diff -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass,InstSimplifyPass" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-DIFF-FILTER-MULT-PASSES
;
; Check that the reporting of IRs respects both -filter-passes and -filter-print-funcs
; RUN: opt -S -print-changed=diff -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass,InstSimplifyPass" -filter-print-funcs=f 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-DIFF-FILTER-FUNC-PASSES
;
; Check that repeated passes that change the IR are printed and that the
; others (including g) are filtered out.  Note that only the first time
; instsimplify is run on f will result in changes
; RUN: opt -S -print-changed=diff -passes="instsimplify,instsimplify" -filter-print-funcs=f  2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-DIFF-MULT-PASSES-FILTER-FUNC
;
; Simple checks of -print-changed=diff-quiet
;
; Note that (mostly) only the banners are checked.
;
; Simple functionality check.
; RUN: opt -S -print-changed=diff-quiet -passes=instsimplify 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-DIFF-QUIET-SIMPLE --allow-empty
;
; Check that only the passes that change the IR are printed and that the
; others (including g) are filtered out.
; RUN: opt -S -print-changed=diff-quiet -passes=instsimplify -filter-print-funcs=f  2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-DIFF-QUIET-FUNC-FILTER
;
; Check that the reporting of IRs respects is not affected by
; -print-module-scope
; RUN: opt -S -print-changed=diff-quiet -passes=instsimplify -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-DIFF-QUIET-PRINT-MOD-SCOPE
;
; Check that reporting of multiple functions happens
; RUN: opt -S -print-changed=diff-quiet -passes=instsimplify -filter-print-funcs="f,g" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-DIFF-QUIET-FILTER-MULT-FUNC
;
; Check that the reporting of IRs respects -filter-passes
; RUN: opt -S -print-changed=diff-quiet -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-DIFF-QUIET-FILTER-PASSES-NONE --allow-empty
;
; Check that the reporting of IRs respects -filter-passes with multiple passes
; RUN: opt -S -print-changed=diff-quiet -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass,InstSimplifyPass" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-DIFF-QUIET-FILTER-MULT-PASSES
;
; Check that the reporting of IRs respects both -filter-passes and -filter-print-funcs
; RUN: opt -S -print-changed=diff-quiet -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass,InstSimplifyPass" -filter-print-funcs=f 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-DIFF-QUIET-FILTER-FUNC-PASSES
;
; Check that repeated passes that change the IR are printed and that the
; others (including g) are filtered out.  Note that only the first time
; instsimplify is run on f will result in changes
; RUN: opt -S -print-changed=diff-quiet -passes="instsimplify,instsimplify" -filter-print-funcs=f  2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-DIFF-QUIET-MULT-PASSES-FILTER-FUNC

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

; CHECK-DIFF-SIMPLE: *** IR Dump At Start: ***
; CHECK-DIFF-SIMPLE: ModuleID = {{.+}}
; CHECK-DIFF-SIMPLE: *** IR Dump After VerifierPass (module) omitted because no change ***
; CHECK-DIFF-SIMPLE: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK-DIFF-SIMPLE-NOT: ModuleID = {{.+}}
; CHECK-DIFF-SIMPLE-NOT: *** IR{{.*}}
; CHECK-DIFF-SIMPLE: entry:
; CHECK-DIFF-SIMPLE-NEXT:-  %a = add i32 2, 3
; CHECK-DIFF-SIMPLE-NEXT:-  ret i32 %a
; CHECK-DIFF-SIMPLE-NEXT:+  ret i32 5
; CHECK-DIFF-SIMPLE: *** IR Pass PassManager{{.*}} (function: g) ignored ***
; CHECK-DIFF-SIMPLE: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-DIFF-SIMPLE-NOT: ModuleID = {{.+}}
; CHECK-DIFF-SIMPLE-NOT: *** IR{{.*}}
; CHECK-DIFF-SIMPLE: entry:
; CHECK-DIFF-SIMPLE-NEXT:-  %a = add i32 2, 3
; CHECK-DIFF-SIMPLE-NEXT:-  ret i32 %a
; CHECK-DIFF-SIMPLE-NEXT:+  ret i32 5
; CHECK-DIFF-SIMPLE: *** IR Pass PassManager{{.*}} (function: f) ignored ***
; CHECK-DIFF-SIMPLE: *** IR Pass ModuleToFunctionPassAdaptor (module) ignored ***
; CHECK-DIFF-SIMPLE: *** IR Dump After VerifierPass (module) omitted because no change ***
; CHECK-DIFF-SIMPLE: *** IR Dump After PrintModulePass (module) omitted because no change ***

; CHECK-DIFF-FUNC-FILTER: *** IR Dump At Start: ***
; CHECK-DIFF-FUNC-FILTER-NEXT: ; ModuleID = {{.+}}
; CHECK-DIFF-FUNC-FILTER: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK-DIFF-FUNC-FILTER: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-DIFF-FUNC-FILTER-NOT: ModuleID = {{.+}}
; CHECK-DIFF-FUNC-FILTER: entry:
; CHECK-DIFF-FUNC-FILTER:-  %a = add i32 2, 3
; CHECK-DIFF-FUNC-FILTER:-  ret i32 %a
; CHECK-DIFF-FUNC-FILTER:+  ret i32 5
; CHECK-DIFF-FUNC-FILTER: *** IR Pass PassManager{{.*}} (function: f) ignored ***
; CHECK-DIFF-FUNC-FILTER: *** IR Pass ModuleToFunctionPassAdaptor (module) ignored ***
; CHECK-DIFF-FUNC-FILTER: *** IR Dump After VerifierPass (module) omitted because no change ***
; CHECK-DIFF-FUNC-FILTER: *** IR Dump After PrintModulePass (module) omitted because no change ***

; CHECK-DIFF-PRINT-MOD-SCOPE: *** IR Dump At Start: ***
; CHECK-DIFF-PRINT-MOD-SCOPE: ModuleID = {{.+}}
; CHECK-DIFF-PRINT-MOD-SCOPE: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK-DIFF-PRINT-MOD-SCOPE-NOT: ModuleID = {{.+}}
; CHECK-DIFF-PRINT-MOD-SCOPE: entry:
; CHECK-DIFF-PRINT-MOD-SCOPE:-  %a = add i32 2, 3
; CHECK-DIFF-PRINT-MOD-SCOPE:-  ret i32 %a
; CHECK-DIFF-PRINT-MOD-SCOPE:+  ret i32 5
; CHECK-DIFF-PRINT-MOD-SCOPE: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-DIFF-PRINT-MOD-SCOPE-NOT: ModuleID = {{.+}}
; CHECK-DIFF-PRINT-MOD-SCOPE: entry:
; CHECK-DIFF-PRINT-MOD-SCOPE:-  %a = add i32 2, 3
; CHECK-DIFF-PRINT-MOD-SCOPE:-  ret i32 %a
; CHECK-DIFF-PRINT-MOD-SCOPE:+  ret i32 5
; CHECK-DIFF-PRINT-MOD-SCOPE: *** IR Pass PassManager{{.*}} (function: f) ignored ***
; CHECK-DIFF-PRINT-MOD-SCOPE: *** IR Pass ModuleToFunctionPassAdaptor (module) ignored ***
; CHECK-DIFF-PRINT-MOD-SCOPE: *** IR Dump After VerifierPass (module) omitted because no change ***
; CHECK-DIFF-PRINT-MOD-SCOPE: *** IR Dump After PrintModulePass (module) omitted because no change ***

; CHECK-DIFF-FILTER-MULT-FUNC: *** IR Dump At Start: ***
; CHECK-DIFF-FILTER-MULT-FUNC: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK-DIFF-FILTER-MULT-FUNC-NOT: ModuleID = {{.+}}
; CHECK-DIFF-FILTER-MULT-FUNC: entry:
; CHECK-DIFF-FILTER-MULT-FUNC:-  %a = add i32 2, 3
; CHECK-DIFF-FILTER-MULT-FUNC:-  ret i32 %a
; CHECK-DIFF-FILTER-MULT-FUNC:+  ret i32 5
; CHECK-DIFF-FILTER-MULT-FUNC: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-DIFF-FILTER-MULT-FUNC-NOT: ModuleID = {{.+}}
; CHECK-DIFF-FILTER-MULT-FUNC: entry:
; CHECK-DIFF-FILTER-MULT-FUNC:-  %a = add i32 2, 3
; CHECK-DIFF-FILTER-MULT-FUNC:-  ret i32 %a
; CHECK-DIFF-FILTER-MULT-FUNC:+  ret i32 5
; CHECK-DIFF-FILTER-MULT-FUNC: *** IR Pass PassManager{{.*}} (function: f) ignored ***
; CHECK-DIFF-FILTER-MULT-FUNC: *** IR Pass ModuleToFunctionPassAdaptor (module) ignored ***
; CHECK-DIFF-FILTER-MULT-FUNC: *** IR Dump After VerifierPass (module) omitted because no change ***
; CHECK-DIFF-FILTER-MULT-FUNC: *** IR Dump After PrintModulePass (module) omitted because no change ***

; CHECK-DIFF-FILTER-PASSES: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK-DIFF-FILTER-PASSES: *** IR Dump At Start: *** (function: g)
; CHECK-DIFF-FILTER-PASSES: *** IR Dump After NoOpFunctionPass (function: g) omitted because no change ***
; CHECK-DIFF-FILTER-PASSES: *** IR Dump After InstSimplifyPass (function: f) filtered out ***
; CHECK-DIFF-FILTER-PASSES: *** IR Dump After NoOpFunctionPass (function: f) omitted because no change ***

; CHECK-DIFF-FILTER-MULT-PASSES: *** IR Dump At Start: *** (function: g)
; CHECK-DIFF-FILTER-MULT-PASSES: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK-DIFF-FILTER-MULT-PASSES-NOT: ModuleID = {{.+}}
; CHECK-DIFF-FILTER-MULT-PASSES: entry:
; CHECK-DIFF-FILTER-MULT-PASSES:-  %a = add i32 2, 3
; CHECK-DIFF-FILTER-MULT-PASSES:-  ret i32 %a
; CHECK-DIFF-FILTER-MULT-PASSES:+  ret i32 5
; CHECK-DIFF-FILTER-MULT-PASSES: *** IR Dump After NoOpFunctionPass (function: g) omitted because no change ***
; CHECK-DIFF-FILTER-MULT-PASSES: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-DIFF-FILTER-MULT-PASSES-NOT: ModuleID = {{.+}}
; CHECK-DIFF-FILTER-MULT-PASSES: entry:
; CHECK-DIFF-FILTER-MULT-PASSES:-  %a = add i32 2, 3
; CHECK-DIFF-FILTER-MULT-PASSES:-  ret i32 %a
; CHECK-DIFF-FILTER-MULT-PASSES:+  ret i32 5
; CHECK-DIFF-FILTER-MULT-PASSES: *** IR Dump After NoOpFunctionPass (function: f) omitted because no change ***

; CHECK-DIFF-FILTER-FUNC-PASSES: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK-DIFF-FILTER-FUNC-PASSES: *** IR Dump After NoOpFunctionPass (function: g) filtered out ***
; CHECK-DIFF-FILTER-FUNC-PASSES: *** IR Dump At Start: *** (function: f)
; CHECK-DIFF-FILTER-FUNC-PASSES: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-DIFF-FILTER-FUNC-PASSES-NOT: ModuleID = {{.+}}
; CHECK-DIFF-FILTER-FUNC-PASSES: entry:
; CHECK-DIFF-FILTER-FUNC-PASSES:-  %a = add i32 2, 3
; CHECK-DIFF-FILTER-FUNC-PASSES:-  ret i32 %a
; CHECK-DIFF-FILTER-FUNC-PASSES:+  ret i32 5
; CHECK-DIFF-FILTER-FUNC-PASSES: *** IR Dump After NoOpFunctionPass (function: f) omitted because no change ***

; CHECK-DIFF-MULT-PASSES-FILTER-FUNC: *** IR Dump At Start: ***
; CHECK-DIFF-MULT-PASSES-FILTER-FUNC: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK-DIFF-MULT-PASSES-FILTER-FUNC: *** IR Dump After InstSimplifyPass (function: g) filtered out ***
; CHECK-DIFF-MULT-PASSES-FILTER-FUNC: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-DIFF-MULT-PASSES-FILTER-FUNC-NOT: ModuleID = {{.+}}
; CHECK-DIFF-MULT-PASSES-FILTER-FUNC: entry:
; CHECK-DIFF-MULT-PASSES-FILTER-FUNC:-  %a = add i32 2, 3
; CHECK-DIFF-MULT-PASSES-FILTER-FUNC:-  ret i32 %a
; CHECK-DIFF-MULT-PASSES-FILTER-FUNC:+  ret i32 5
; CHECK-DIFF-MULT-PASSES-FILTER-FUNC: *** IR Dump After InstSimplifyPass (function: f) omitted because no change ***

; CHECK-DIFF-QUIET-SIMPLE-NOT: *** IR Dump {{.*(At Start:|no change|ignored|filtered out)}} ***
; CHECK-DIFF-QUIET-SIMPLE: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK-DIFF-QUIET-SIMPLE-NOT: ModuleID = {{.+}}
; CHECK-DIFF-QUIET-SIMPLE-NOT: *** IR{{.*}}
; CHECK-DIFF-QUIET-SIMPLE: entry:
; CHECK-DIFF-QUIET-SIMPLE-NEXT:-  %a = add i32 2, 3
; CHECK-DIFF-QUIET-SIMPLE-NEXT:-  ret i32 %a
; CHECK-DIFF-QUIET-SIMPLE-NEXT:+  ret i32 5
; CHECK-DIFF-QUIET-SIMPLE-EMPTY:
; CHECK-DIFF-QUIET-SIMPLE-NEXT: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-DIFF-QUIET-SIMPLE-NOT: ModuleID = {{.+}}
; CHECK-DIFF-QUIET-SIMPLE-NOT: *** IR{{.*}}
; CHECK-DIFF-QUIET-SIMPLE: entry:
; CHECK-DIFF-QUIET-SIMPLE-NEXT:-  %a = add i32 2, 3
; CHECK-DIFF-QUIET-SIMPLE-NEXT:-  ret i32 %a
; CHECK-DIFF-QUIET-SIMPLE-NEXT:+  ret i32 5
; CHECK-DIFF-QUIET-SIMPLE-NOT: *** IR{{.*}}

; CHECK-DIFF-QUIET-FUNC-FILTER-NOT: *** IR Dump {{.*(At Start:|no change|ignored|filtered out)}} ***
; CHECK-DIFF-QUIET-FUNC-FILTER: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-DIFF-QUIET-FUNC-FILTER-NOT: ModuleID = {{.+}}
; CHECK-DIFF-QUIET-FUNC-FILTER: entry:
; CHECK-DIFF-QUIET-FUNC-FILTER:-  %a = add i32 2, 3
; CHECK-DIFF-QUIET-FUNC-FILTER:-  ret i32 %a
; CHECK-DIFF-QUIET-FUNC-FILTER:+  ret i32 5
; CHECK-DIFF-QUIET-FUNC-FILTER-NOT: *** IR{{.*}}

; CHECK-DIFF-QUIET-PRINT-MOD-SCOPE-NOT: *** IR Dump {{.*(At Start:|no change|ignored|filtered out)}} ***
; CHECK-DIFF-QUIET-PRINT-MOD-SCOPE: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK-DIFF-QUIET-PRINT-MOD-SCOPE-NOT: ModuleID = {{.+}}
; CHECK-DIFF-QUIET-PRINT-MOD-SCOPE: entry:
; CHECK-DIFF-QUIET-PRINT-MOD-SCOPE:-  %a = add i32 2, 3
; CHECK-DIFF-QUIET-PRINT-MOD-SCOPE:-  ret i32 %a
; CHECK-DIFF-QUIET-PRINT-MOD-SCOPE:+  ret i32 5
; CHECK-DIFF-QUIET-PRINT-MOD-SCOPE-EMPTY:
; CHECK-DIFF-QUIET-PRINT-MOD-SCOPE-NEXT: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-DIFF-QUIET-PRINT-MOD-SCOPE-NOT: ModuleID = {{.+}}
; CHECK-DIFF-QUIET-PRINT-MOD-SCOPE: entry:
; CHECK-DIFF-QUIET-PRINT-MOD-SCOPE:-  %a = add i32 2, 3
; CHECK-DIFF-QUIET-PRINT-MOD-SCOPE:-  ret i32 %a
; CHECK-DIFF-QUIET-PRINT-MOD-SCOPE:+  ret i32 5
; CHECK-DIFF-QUIET-PRINT-MOD-SCOPE-NOT: *** IR{{.*}}

; CHECK-DIFF-QUIET-FILTER-MULT-FUNC-NOT: *** IR Dump {{.*(At Start:|no change|ignored|filtered out)}} ***
; CHECK-DIFF-QUIET-FILTER-MULT-FUNC: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK-DIFF-QUIET-FILTER-MULT-FUNC-NOT: ModuleID = {{.+}}
; CHECK-DIFF-QUIET-FILTER-MULT-FUNC: entry:
; CHECK-DIFF-QUIET-FILTER-MULT-FUNC:-  %a = add i32 2, 3
; CHECK-DIFF-QUIET-FILTER-MULT-FUNC:-  ret i32 %a
; CHECK-DIFF-QUIET-FILTER-MULT-FUNC:+  ret i32 5
; CHECK-DIFF-QUIET-FILTER-MULT-FUNC-EMPTY:
; CHECK-DIFF-QUIET-FILTER-MULT-FUNC-NEXT: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-DIFF-QUIET-FILTER-MULT-FUNC-NOT: ModuleID = {{.+}}
; CHECK-DIFF-QUIET-FILTER-MULT-FUNC: entry:
; CHECK-DIFF-QUIET-FILTER-MULT-FUNC:-  %a = add i32 2, 3
; CHECK-DIFF-QUIET-FILTER-MULT-FUNC:-  ret i32 %a
; CHECK-DIFF-QUIET-FILTER-MULT-FUNC:+  ret i32 5
; CHECK-DIFF-QUIET-FILTER-MULT-FUNC-NOT: *** IR{{.*}}

; CHECK-DIFF-QUIET-FILTER-PASSES-NONE-NOT: *** IR

; CHECK-DIFF-QUIET-FILTER-MULT-PASSES-NOT: *** IR Dump {{.*(At Start:|no change|ignored|filtered out)}} ***
; CHECK-DIFF-QUIET-FILTER-MULT-PASSES: *** IR Dump After InstSimplifyPass *** (function: g)
; CHECK-DIFF-QUIET-FILTER-MULT-PASSES-NOT: ModuleID = {{.+}}
; CHECK-DIFF-QUIET-FILTER-MULT-PASSES: entry:
; CHECK-DIFF-QUIET-FILTER-MULT-PASSES:-  %a = add i32 2, 3
; CHECK-DIFF-QUIET-FILTER-MULT-PASSES:-  ret i32 %a
; CHECK-DIFF-QUIET-FILTER-MULT-PASSES:+  ret i32 5
; CHECK-DIFF-QUIET-FILTER-MULT-PASSES-EMPTY:
; CHECK-DIFF-QUIET-FILTER-MULT-PASSES: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-DIFF-QUIET-FILTER-MULT-PASSES-NOT: ModuleID = {{.+}}
; CHECK-DIFF-QUIET-FILTER-MULT-PASSES: entry:
; CHECK-DIFF-QUIET-FILTER-MULT-PASSES:-  %a = add i32 2, 3
; CHECK-DIFF-QUIET-FILTER-MULT-PASSES:-  ret i32 %a
; CHECK-DIFF-QUIET-FILTER-MULT-PASSES:+  ret i32 5
; CHECK-DIFF-QUIET-FILTER-MULT-PASSES-NOT: *** IR

; CHECK-DIFF-QUIET-FILTER-FUNC-PASSES-NOT: *** IR Dump {{.*(At Start:|no change|ignored|filtered out)}} ***
; CHECK-DIFF-QUIET-FILTER-FUNC-PASSES: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-DIFF-QUIET-FILTER-FUNC-PASSES-NOT: ModuleID = {{.+}}
; CHECK-DIFF-QUIET-FILTER-FUNC-PASSES: entry:
; CHECK-DIFF-QUIET-FILTER-FUNC-PASSES:-  %a = add i32 2, 3
; CHECK-DIFF-QUIET-FILTER-FUNC-PASSES:-  ret i32 %a
; CHECK-DIFF-QUIET-FILTER-FUNC-PASSES:+  ret i32 5
; CHECK-DIFF-QUIET-FILTER-FUNC-PASSES-NOT: *** IR

; CHECK-DIFF-QUIET-MULT-PASSES-FILTER-FUNC-NOT: *** IR Dump {{.*(At Start:|no change|ignored|filtered out)}} ***
; CHECK-DIFF-QUIET-MULT-PASSES-FILTER-FUNC: *** IR Dump After InstSimplifyPass *** (function: f)
; CHECK-DIFF-QUIET-MULT-PASSES-FILTER-FUNC-NOT: ModuleID = {{.+}}
; CHECK-DIFF-QUIET-MULT-PASSES-FILTER-FUNC: entry:
; CHECK-DIFF-QUIET-MULT-PASSES-FILTER-FUNC:-  %a = add i32 2, 3
; CHECK-DIFF-QUIET-MULT-PASSES-FILTER-FUNC:-  ret i32 %a
; CHECK-DIFF-QUIET-MULT-PASSES-FILTER-FUNC:+  ret i32 5
; CHECK-DIFF-QUIET-MULT-PASSES-FILTER-FUNC-NOT: *** IR
