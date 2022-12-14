; Simple checks of -print-changed functionality
;
; Note that (mostly) only the banners are checked.
;
; Simple functionality check.
; RUN: opt -S -print-changed -passes=instsimplify 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-SIMPLE
;
; Simple functionality check.
; RUN: opt -S -print-changed= -passes=instsimplify 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-SIMPLE
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
; RUN: opt -S -print-changed -passes="instsimplify,no-op-function" -filter-passes="no-op-function" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FILTER-PASSES
;
; Check that the reporting of IRs respects -filter-passes with multiple passes
; RUN: opt -S -print-changed -passes="instsimplify,no-op-function" -filter-passes="no-op-function,instsimplify" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FILTER-MULT-PASSES
;
; Check that the reporting of IRs respects both -filter-passes and -filter-print-funcs
; RUN: opt -S -print-changed -passes="instsimplify,no-op-function" -filter-passes="no-op-function,instsimplify" -filter-print-funcs=f 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FILTER-FUNC-PASSES
;
; Check that the reporting of IRs respects -filter-passes, -filter-print-funcs and -print-module-scope
; RUN: opt -S -print-changed -passes="instsimplify,no-op-function" -filter-passes="no-op-function,instsimplify" -filter-print-funcs=f -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FILTER-FUNC-PASSES-MOD-SCOPE
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
; Simple checks of -print-changed=quiet functionality
;
; Simple functionality check.
; RUN: opt -S -print-changed=quiet -passes=instsimplify 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-QUIET-SIMPLE
;
; Check that only the passes that change the IR are printed and that the
; others (including g) are filtered out.
; RUN: opt -S -print-changed=quiet -passes=instsimplify -filter-print-funcs=f  2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-QUIET-FUNC-FILTER
;
; Check that the reporting of IRs respects -print-module-scope
; RUN: opt -S -print-changed=quiet -passes=instsimplify -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-QUIET-PRINT-MOD-SCOPE
;
; Check that the reporting of IRs respects -print-module-scope
; RUN: opt -S -print-changed=quiet -passes=instsimplify -filter-print-funcs=f -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-QUIET-FUNC-FILTER-MOD-SCOPE
;
; Check that reporting of multiple functions happens
; RUN: opt -S -print-changed=quiet -passes=instsimplify -filter-print-funcs="f,g" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-QUIET-FILTER-MULT-FUNC
;
; Check that the reporting of IRs respects -filter-passes
; RUN: opt -S -print-changed=quiet -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-QUIET-FILTER-PASSES-NONE --allow-empty
;
; Check that the reporting of IRs respects -filter-passes with multiple passes
; RUN: opt -S -print-changed=quiet -passes="instsimplify" -filter-passes="no-op-function,instsimplify" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-QUIET-FILTER-PASSES
;
; Check that the reporting of IRs respects -filter-passes with multiple passes
; RUN: opt -S -print-changed=quiet -passes="instsimplify,no-op-function" -filter-passes="no-op-function,instsimplify" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-QUIET-FILTER-MULT-PASSES
;
; Check that the reporting of IRs respects both -filter-passes and -filter-print-funcs
; RUN: opt -S -print-changed=quiet -passes="instsimplify,no-op-function" -filter-passes="no-op-function,instsimplify" -filter-print-funcs=f 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-QUIET-FILTER-FUNC-PASSES
;
; Check that the reporting of IRs respects -filter-passes, -filter-print-funcs and -print-module-scope
; RUN: opt -S -print-changed=quiet -passes="instsimplify,no-op-function" -filter-passes="no-op-function,instsimplify" -filter-print-funcs=f -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-QUIET-FILTER-FUNC-PASSES-MOD-SCOPE
;
; Check that repeated passes that change the IR are printed and that the
; others (including g) are filtered out.  Note that the second time
; instsimplify is run on f, it does not change the IR
; RUN: opt -S -print-changed=quiet -passes="instsimplify,instsimplify" -filter-print-funcs=f  2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-QUIET-MULT-PASSES-FILTER-FUNC
;
; Simple print-before-changed functionality check.
; RUN: opt -S -print-changed=quiet -print-before-changed -passes=instsimplify 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-SIMPLE-BEFORE-QUIET
;
; Check print-before-changed obeys the function filtering
; RUN: opt -S -print-changed=quiet -print-before-changed -passes=instsimplify -filter-print-funcs=f  2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FUNC-FILTER-BEFORE-QUIET
;
; Check that the reporting of IRs with -print-before-changed respects -print-module-scope
; RUN: opt -S -print-changed=quiet -print-before-changed -passes=instsimplify -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-PRINT-MOD-SCOPE-BEFORE-QUIET
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

; CHECK-SIMPLE: *** IR Dump At Start ***
; CHECK-SIMPLE-NEXT: ; ModuleID = {{.+}}
; CHECK-SIMPLE: *** IR Dump After InstSimplifyPass on g ***
; CHECK-SIMPLE-NEXT: define i32 @g()
; CHECK-SIMPLE: *** IR Pass PassManager{{.*}} on g ignored ***
; CHECK-SIMPLE: *** IR Dump After InstSimplifyPass on f ***
; CHECK-SIMPLE-NEXT: define i32 @f()
; CHECK-SIMPLE: *** IR Pass PassManager{{.*}} on f ignored ***
; CHECK-SIMPLE: *** IR Pass ModuleToFunctionPassAdaptor on [module] ignored ***
; CHECK-SIMPLE: *** IR Dump After VerifierPass on [module] omitted because no change ***
; CHECK-SIMPLE: *** IR Dump After PrintModulePass on [module] omitted because no change ***
; CHECK-SIMPLE-NOT: *** IR

; CHECK-FUNC-FILTER: *** IR Dump At Start ***
; CHECK-FUNC-FILTER-NEXT: ; ModuleID = {{.+}}
; CHECK-FUNC-FILTER: *** IR Dump After InstSimplifyPass on g filtered out ***
; CHECK-FUNC-FILTER: *** IR Dump After InstSimplifyPass on f ***
; CHECK-FUNC-FILTER-NEXT: define i32 @f()

; CHECK-PRINT-MOD-SCOPE: *** IR Dump At Start ***
; CHECK-PRINT-MOD-SCOPE-NEXT: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE: *** IR Dump After InstSimplifyPass on g ***
; CHECK-PRINT-MOD-SCOPE-NEXT: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE: *** IR Dump After InstSimplifyPass on f ***
; CHECK-PRINT-MOD-SCOPE-NEXT: ModuleID = {{.+}}

; CHECK-FUNC-FILTER-MOD-SCOPE: *** IR Dump At Start ***
; CHECK-FUNC-FILTER-MOD-SCOPE-NEXT: ; ModuleID = {{.+}}
; CHECK-FUNC-FILTER-MOD-SCOPE: *** IR Dump After InstSimplifyPass on g filtered out ***
; CHECK-FUNC-FILTER-MOD-SCOPE: *** IR Dump After InstSimplifyPass on f ***
; CHECK-FUNC-FILTER-MOD-SCOPE-NEXT: ModuleID = {{.+}}

; CHECK-FILTER-MULT-FUNC: *** IR Dump At Start ***
; CHECK-FILTER-MULT-FUNC-NEXT: ; ModuleID = {{.+}}
; CHECK-FILTER-MULT-FUNC: *** IR Dump After InstSimplifyPass on g ***
; CHECK-FILTER-MULT-FUNC-NEXT: define i32 @g()
; CHECK-FILTER-MULT-FUNC: *** IR Dump After InstSimplifyPass on f ***
; CHECK-FILTER-MULT-FUNC-NEXT: define i32 @f()

; CHECK-FILTER-PASSES: *** IR Dump At Start ***
; CHECK-FILTER-PASSES-NEXT: ; ModuleID = {{.+}}
; CHECK-FILTER-PASSES: *** IR Dump After InstSimplifyPass on g filtered out ***
; CHECK-FILTER-PASSES: *** IR Dump After NoOpFunctionPass on g omitted because no change ***
; CHECK-FILTER-PASSES: *** IR Dump After InstSimplifyPass on f filtered out ***
; CHECK-FILTER-PASSES: *** IR Dump After NoOpFunctionPass on f omitted because no change ***

; CHECK-FILTER-MULT-PASSES: *** IR Dump At Start ***
; CHECK-FILTER-MULT-PASSES-NEXT: ; ModuleID = {{.+}}
; CHECK-FILTER-MULT-PASSES: *** IR Dump After InstSimplifyPass on g ***
; CHECK-FILTER-MULT-PASSES-NEXT: define i32 @g()
; CHECK-FILTER-MULT-PASSES: *** IR Dump After NoOpFunctionPass on g omitted because no change ***
; CHECK-FILTER-MULT-PASSES: *** IR Dump After InstSimplifyPass on f ***
; CHECK-FILTER-MULT-PASSES-NEXT: define i32 @f()
; CHECK-FILTER-MULT-PASSES: *** IR Dump After NoOpFunctionPass on f omitted because no change ***

; CHECK-FILTER-FUNC-PASSES: *** IR Dump At Start ***
; CHECK-FILTER-FUNC-PASSES-NEXT: ; ModuleID = {{.+}}
; CHECK-FILTER-FUNC-PASSES: *** IR Dump After InstSimplifyPass on g filtered out ***
; CHECK-FILTER-FUNC-PASSES: *** IR Dump After NoOpFunctionPass on g filtered out ***
; CHECK-FILTER-FUNC-PASSES: *** IR Dump After InstSimplifyPass on f ***
; CHECK-FILTER-FUNC-PASSES-NEXT: define i32 @f()
; CHECK-FILTER-FUNC-PASSES: *** IR Dump After NoOpFunctionPass on f omitted because no change ***

; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE: *** IR Dump At Start ***
; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE-NEXT: ; ModuleID = {{.+}}
; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE: *** IR Dump After InstSimplifyPass on g filtered out ***
; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE: *** IR Dump After NoOpFunctionPass on g filtered out ***
; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE: *** IR Dump After InstSimplifyPass on f ***
; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE-NEXT: ModuleID = {{.+}}
; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE: *** IR Dump After NoOpFunctionPass on f omitted because no change ***

; CHECK-MULT-PASSES-FILTER-FUNC: *** IR Dump At Start ***
; CHECK-MULT-PASSES-FILTER-FUNC-NEXT: ; ModuleID = {{.+}}
; CHECK-MULT-PASSES-FILTER-FUNC: *** IR Dump After InstSimplifyPass on g filtered out ***
; CHECK-MULT-PASSES-FILTER-FUNC: *** IR Dump After InstSimplifyPass on g filtered out ***
; CHECK-MULT-PASSES-FILTER-FUNC: *** IR Dump After InstSimplifyPass on f ***
; CHECK-MULT-PASSES-FILTER-FUNC-NEXT: define i32 @f()
; CHECK-MULT-PASSES-FILTER-FUNC: *** IR Dump After InstSimplifyPass on f omitted because no change ***

; CHECK-SIMPLE-BEFORE: *** IR Dump At Start ***
; CHECK-SIMPLE-BEFORE-NEXT: ; ModuleID = {{.+}}
; CHECK-SIMPLE-BEFORE: *** IR Dump Before InstSimplifyPass on g ***
; CHECK-SIMPLE-BEFORE-NEXT: define i32 @g()
; CHECK-SIMPLE-BEFORE: *** IR Dump After InstSimplifyPass on g ***
; CHECK-SIMPLE-BEFORE-NEXT: define i32 @g()
; CHECK-SIMPLE-BEFORE: *** IR Pass PassManager{{.*}} on g ignored ***
; CHECK-SIMPLE-BEFORE: *** IR Dump Before InstSimplifyPass on f ***
; CHECK-SIMPLE-BEFORE-NEXT: define i32 @f()
; CHECK-SIMPLE-BEFORE: *** IR Dump After InstSimplifyPass on f ***
; CHECK-SIMPLE-BEFORE-NEXT: define i32 @f()
; CHECK-SIMPLE-BEFORE: *** IR Pass PassManager{{.*}} on f ignored ***
; CHECK-SIMPLE-BEFORE: *** IR Pass ModuleToFunctionPassAdaptor on [module] ignored ***
; CHECK-SIMPLE-BEFORE: *** IR Dump After VerifierPass on [module] omitted because no change ***
; CHECK-SIMPLE-BEFORE: *** IR Dump After PrintModulePass on [module] omitted because no change ***
; CHECK-SIMPLE-BEFORE-NOT: *** IR

; CHECK-FUNC-FILTER-BEFORE: *** IR Dump At Start ***
; CHECK-FUNC-FILTER-BEFORE-NEXT: ; ModuleID = {{.+}}
; CHECK-FUNC-FILTER-BEFORE: *** IR Dump After InstSimplifyPass on g filtered out ***
; CHECK-FUNC-FILTER-BEFORE: *** IR Dump Before InstSimplifyPass on f ***
; CHECK-FUNC-FILTER-BEFORE-NEXT: define i32 @f()
; CHECK-FUNC-FILTER-BEFORE: *** IR Dump After InstSimplifyPass on f ***
; CHECK-FUNC-FILTER-BEFORE-NEXT: define i32 @f()

; CHECK-PRINT-MOD-SCOPE-BEFORE: *** IR Dump At Start ***
; CHECK-PRINT-MOD-SCOPE-BEFORE-NEXT: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE-BEFORE: *** IR Dump Before InstSimplifyPass on g ***
; CHECK-PRINT-MOD-SCOPE-BEFORE-NEXT: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE-BEFORE: *** IR Dump After InstSimplifyPass on g ***
; CHECK-PRINT-MOD-SCOPE-BEFORE-NEXT: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE-BEFORE: *** IR Dump Before InstSimplifyPass on f ***
; CHECK-PRINT-MOD-SCOPE-BEFORE-NEXT: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE-BEFORE: *** IR Dump After InstSimplifyPass on f ***
; CHECK-PRINT-MOD-SCOPE-BEFORE-NEXT: ModuleID = {{.+}}

; CHECK-QUIET-SIMPLE-NOT: *** IR Dump {{.*(At Start|no change|ignored|filtered out)}} ***
; CHECK-QUIET-SIMPLE: *** IR Dump After InstSimplifyPass on g ***
; CHECK-QUIET-SIMPLE-NEXT: define i32 @g()
; CHECK-QUIET-SIMPLE-NOT: *** IR Dump {{.*(no change|ignored|filtered out)}} ***
; CHECK-QUIET-SIMPLE: *** IR Dump After InstSimplifyPass on f ***
; CHECK-QUIET-SIMPLE-NEXT: define i32 @f()
; CHECK-QUIET-SIMPLE-NOT: *** IR

; CHECK-QUIET-FUNC-FILTER-NOT: *** IR Dump {{.*(At Start|no change|ignored|filtered out)}} ***
; CHECK-QUIET-FUNC-FILTER: *** IR Dump After InstSimplifyPass on f ***
; CHECK-QUIET-FUNC-FILTER-NEXT: define i32 @f()
; CHECK-QUIET-FUNC-FILTER-NOT: *** IR

; CHECK-QUIET-PRINT-MOD-SCOPE-NOT: *** IR Dump {{.*(At Start|no change|ignored|filtered out)}} ***
; CHECK-QUIET-PRINT-MOD-SCOPE: *** IR Dump After InstSimplifyPass on g ***
; CHECK-QUIET-PRINT-MOD-SCOPE-NEXT: ModuleID = {{.+}}
; CHECK-QUIET-PRINT-MOD-SCOPE: *** IR Dump After InstSimplifyPass on f ***
; CHECK-QUIET-PRINT-MOD-SCOPE-NEXT: ModuleID = {{.+}}
; CHECK-QUIET-PRINT-MOD-SCOPE-NOT: *** IR

; CHECK-QUIET-FUNC-FILTER-MOD-SCOPE-NOT: *** IR Dump {{.*(At Start|no change|ignored|filtered out)}} ***
; CHECK-QUIET-FUNC-FILTER-MOD-SCOPE: *** IR Dump After InstSimplifyPass on f ***
; CHECK-QUIET-FUNC-FILTER-MOD-SCOPE-NEXT: ModuleID = {{.+}}
; CHECK-QUIET-FUNC-FILTER-MOD-SCOPE-NOT: *** IR

; CHECK-QUIET-FILTER-MULT-FUNC-NOT: *** IR Dump {{.*(At Start|no change|ignored|filtered out)}} ***
; CHECK-QUIET-FILTER-MULT-FUNC: *** IR Dump After InstSimplifyPass on g ***
; CHECK-QUIET-FILTER-MULT-FUNC-NEXT: define i32 @g()
; CHECK-QUIET-FILTER-MULT-FUNC: *** IR Dump After InstSimplifyPass on f ***
; CHECK-QUIET-FILTER-MULT-FUNC-NEXT: define i32 @f()
; CHECK-QUIET-FILTER-MULT-FUNC-NOT: *** IR

; CHECK-QUIET-FILTER-PASSES-NONE-NOT: *** IR

; CHECK-QUIET-FILTER-PASSES-NOT: *** IR Dump {{.*(At Start|no change|ignored|filtered out)}} ***
; CHECK-QUIET-FILTER-PASSES: *** IR Dump After InstSimplifyPass on g ***
; CHECK-QUIET-FILTER-PASSES-NEXT: define i32 @g()
; CHECK-QUIET-FILTER-PASSES-NOT: *** IR Dump {{.*(At Start|no change|ignored|filtered out)}} ***
; CHECK-QUIET-FILTER-PASSES: *** IR Dump After InstSimplifyPass on f ***
; CHECK-QUIET-FILTER-PASSES-NEXT: define i32 @f()
; CHECK-QUIET-FILTER-PASSES-NOT: *** IR

; CHECK-QUIET-FILTER-MULT-PASSES-NOT: *** IR Dump {{.*(At Start|no change|ignored|filtered out)}} ***
; CHECK-QUIET-FILTER-MULT-PASSES: *** IR Dump After InstSimplifyPass on g ***
; CHECK-QUIET-FILTER-MULT-PASSES-NEXT: define i32 @g()
; CHECK-QUIET-FILTER-MULT-PASSES-NOT: *** IR Dump {{.*(At Start|no change|ignored|filtered out)}} ***
; CHECK-QUIET-FILTER-MULT-PASSES: *** IR Dump After InstSimplifyPass on f ***
; CHECK-QUIET-FILTER-MULT-PASSES-NEXT: define i32 @f()
; CHECK-QUIET-FILTER-MULT-PASSES-NOT: *** IR

; CHECK-QUIET-FILTER-FUNC-PASSES-NOT: *** IR Dump {{.*(At Start|no change|ignored|filtered out)}} ***
; CHECK-QUIET-FILTER-FUNC-PASSES: *** IR Dump After InstSimplifyPass on f ***
; CHECK-QUIET-FILTER-FUNC-PASSES-NEXT: define i32 @f()
; CHECK-QUIET-FILTER-FUNC-PASSES-NOT: *** IR

; CHECK-QUIET-FILTER-FUNC-PASSES-MOD-SCOPE-NOT: *** IR Dump {{.*(At Start|no change|ignored|filtered out)}} ***
; CHECK-QUIET-FILTER-FUNC-PASSES-MOD-SCOPE: *** IR Dump After InstSimplifyPass on f ***
; CHECK-QUIET-FILTER-FUNC-PASSES-MOD-SCOPE-NEXT: ModuleID = {{.+}}
; CHECK-QUIET-FILTER-FUNC-PASSES-MOD-SCOPE-NOT: *** IR

; CHECK-QUIET-MULT-PASSES-FILTER-FUNC-NOT: *** IR Dump {{.*(At Start|no change|ignored|filtered out)}} ***
; CHECK-QUIET-MULT-PASSES-FILTER-FUNC: *** IR Dump After InstSimplifyPass on f ***
; CHECK-QUIET-MULT-PASSES-FILTER-FUNC-NEXT: define i32 @f()
; CHECK-QUIET-MULT-PASSES-FILTER-FUNC-NOT: *** IR

; CHECK-SIMPLE-BEFORE-QUIET-NOT: *** IR Dump {{.*(At Start|no change|ignored|filtered out)}} ***
; CHECK-SIMPLE-BEFORE-QUIET: *** IR Dump Before InstSimplifyPass on g ***
; CHECK-SIMPLE-BEFORE-QUIET-NEXT: define i32 @g()
; CHECK-SIMPLE-BEFORE-QUIET-NOT: *** IR Dump {{.*(no change|ignored|filtered out)}} ***
; CHECK-SIMPLE-BEFORE-QUIET: *** IR Dump After InstSimplifyPass on g ***
; CHECK-SIMPLE-BEFORE-QUIET-NEXT: define i32 @g()
; CHECK-SIMPLE-BEFORE-QUIET-NOT: *** IR Dump {{.*(no change|ignored|filtered out)}} ***
; CHECK-SIMPLE-BEFORE-QUIET: *** IR Dump Before InstSimplifyPass on f ***
; CHECK-SIMPLE-BEFORE-QUIET-NEXT: define i32 @f()
; CHECK-SIMPLE-BEFORE-QUIET-NOT: *** IR Dump {{.*(no change|ignored|filtered out)}} ***
; CHECK-SIMPLE-BEFORE-QUIET: *** IR Dump After InstSimplifyPass on f ***
; CHECK-SIMPLE-BEFORE-QUIET-NEXT: define i32 @f()
; CHECK-SIMPLE-BEFORE-QUIET-NOT: *** IR

; CHECK-FUNC-FILTER-BEFORE-QUIET-NOT: *** IR Dump {{.*(At Start|no change|ignored|filtered out)}} ***
; CHECK-FUNC-FILTER-BEFORE-QUIET: *** IR Dump Before InstSimplifyPass on f ***
; CHECK-FUNC-FILTER-BEFORE-QUIET-NEXT: define i32 @f()
; CHECK-FUNC-FILTER-BEFORE-QUIET-NOT: *** IR Dump {{.*(no change|ignored|filtered out)}} ***
; CHECK-FUNC-FILTER-BEFORE-QUIET: *** IR Dump After InstSimplifyPass on f ***
; CHECK-FUNC-FILTER-BEFORE-QUIET-NEXT: define i32 @f()
; CHECK-FUNC-FILTER-BEFORE-QUIET-NOT: *** IR

; CHECK-PRINT-MOD-SCOPE-BEFORE-QUIET-NOT: *** IR Dump {{.*(At Start|no change|ignored|filtered out)}} ***
; CHECK-PRINT-MOD-SCOPE-BEFORE-QUIET: *** IR Dump Before InstSimplifyPass on g ***
; CHECK-PRINT-MOD-SCOPE-BEFORE-QUIET-NEXT: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE-BEFORE-QUIET-NOT: *** IR Dump {{.*(no change|ignored|filtered out)}} ***
; CHECK-PRINT-MOD-SCOPE-BEFORE-QUIET: *** IR Dump After InstSimplifyPass on g ***
; CHECK-PRINT-MOD-SCOPE-BEFORE-QUIET-NEXT: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE-BEFORE-QUIET-NOT: *** IR Dump {{.*(no change|ignored|filtered out)}} ***
; CHECK-PRINT-MOD-SCOPE-BEFORE-QUIET: *** IR Dump Before InstSimplifyPass on f ***
; CHECK-PRINT-MOD-SCOPE-BEFORE-QUIET-NEXT: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE-BEFORE-QUIET-NOT: *** IR Dump {{.*(no change|ignored|filtered out)}} ***
; CHECK-PRINT-MOD-SCOPE-BEFORE-QUIET: *** IR Dump After InstSimplifyPass on f ***
; CHECK-PRINT-MOD-SCOPE-BEFORE-QUIET-NEXT: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE-BEFORE-QUIET-NOT: *** IR
