; Simple checks of -test-changed=%S/test-changed-script.sh functionality
;
; Simple functionality check.
; RUN: opt -S -test-changed=%S/test-changed-script.sh -passes=instsimplify 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-SIMPLE
;
; Check that only the passes that change the IR are printed and that the
; others (including g) are filtered out.
; RUN: opt -S -test-changed=%S/test-changed-script.sh -passes=instsimplify -filter-print-funcs=f  2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FUNC-FILTER
;
; Check that the reporting of IRs respects -print-module-scope
; RUN: opt -S -test-changed=%S/test-changed-script.sh -passes=instsimplify -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-PRINT-MOD-SCOPE
;
; Check that the reporting of IRs respects -print-module-scope
; RUN: opt -S -test-changed=%S/test-changed-script.sh -passes=instsimplify -filter-print-funcs=f -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FUNC-FILTER-MOD-SCOPE
;
; Check that reporting of multiple functions happens
; RUN: opt -S -test-changed=%S/test-changed-script.sh -passes=instsimplify -filter-print-funcs="f,g" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FILTER-MULT-FUNC
;
; Check that the reporting of IRs respects -filter-passes
; RUN: opt -S -test-changed=%S/test-changed-script.sh -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FILTER-PASSES
;
; Check that the reporting of IRs respects -filter-passes with multiple passes
; RUN: opt -S -test-changed=%S/test-changed-script.sh -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass,InstSimplifyPass" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FILTER-MULT-PASSES
;
; Check that the reporting of IRs respects both -filter-passes and -filter-print-funcs
; RUN: opt -S -test-changed=%S/test-changed-script.sh -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass,InstSimplifyPass" -filter-print-funcs=f 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FILTER-FUNC-PASSES
;
; Check that the reporting of IRs respects -filter-passes, -filter-print-funcs and -print-module-scope
; RUN: opt -S -test-changed=%S/test-changed-script.sh -passes="instsimplify,no-op-function" -filter-passes="NoOpFunctionPass,InstSimplifyPass" -filter-print-funcs=f -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FILTER-FUNC-PASSES-MOD-SCOPE
;
; Check that repeated passes that change the IR are printed and that the
; others (including g) are filtered out.  Note that the second time
; instsimplify is run on f, it does not change the IR
; RUN: opt -S -test-changed=%S/test-changed-script.sh -passes="instsimplify,instsimplify" -filter-print-funcs=f  2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-MULT-PASSES-FILTER-FUNC
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

; CHECK-SIMPLE: *** Initial IR ***
; CHECK-SIMPLE-NEXT: ; ModuleID = {{.+}}
; CHECK-SIMPLE: *** InstSimplifyPass ***
; CHECK-SIMPLE-NEXT: define i32 @g()
; CHECK-SIMPLE: *** InstSimplifyPass ***
; CHECK-SIMPLE-NEXT: define i32 @f()

; CHECK-FUNC-FILTER: *** Initial IR ***
; CHECK-FUNC-FILTER-NEXT: define i32 @f()
; CHECK-FUNC-FILTER: *** InstSimplifyPass ***
; CHECK-FUNC-FILTER-NEXT: define i32 @f()

; CHECK-PRINT-MOD-SCOPE: *** Initial IR ***
; CHECK-PRINT-MOD-SCOPE-NEXT: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE: *** InstSimplifyPass ***
; CHECK-PRINT-MOD-SCOPE-NEXT: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE: *** InstSimplifyPass ***
; CHECK-PRINT-MOD-SCOPE-NEXT: ModuleID = {{.+}}

; CHECK-FUNC-FILTER-MOD-SCOPE: *** Initial IR ***
; CHECK-FUNC-FILTER-MOD-SCOPE-NEXT: ; ModuleID = {{.+}}
; CHECK-FUNC-FILTER-MOD-SCOPE: *** InstSimplifyPass ***
; CHECK-FUNC-FILTER-MOD-SCOPE-NEXT: ModuleID = {{.+}}

; CHECK-FILTER-MULT-FUNC: *** Initial IR ***
; CHECK-FILTER-MULT-FUNC-NEXT: define i32 @g()
; CHECK-FILTER-MULT-FUNC: *** InstSimplifyPass ***
; CHECK-FILTER-MULT-FUNC-NEXT: define i32 @g()
; CHECK-FILTER-MULT-FUNC: *** InstSimplifyPass ***
; CHECK-FILTER-MULT-FUNC-NEXT: define i32 @f()

; CHECK-FILTER-PASSES: *** Initial IR ***
; CHECK-FILTER-PASSES-NEXT: define i32 @g()

; CHECK-FILTER-MULT-PASSES: *** Initial IR ***
; CHECK-FILTER-MULT-PASSES-NEXT: define i32 @g()
; CHECK-FILTER-MULT-PASSES: *** InstSimplifyPass ***
; CHECK-FILTER-MULT-PASSES-NEXT: define i32 @g()
; CHECK-FILTER-MULT-PASSES: *** InstSimplifyPass ***
; CHECK-FILTER-MULT-PASSES-NEXT: define i32 @f()

; CHECK-FILTER-FUNC-PASSES: *** Initial IR ***
; CHECK-FILTER-FUNC-PASSES-NEXT: define i32 @f()
; CHECK-FILTER-FUNC-PASSES: *** InstSimplifyPass ***
; CHECK-FILTER-FUNC-PASSES-NEXT: define i32 @f()

; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE: *** Initial IR ***
; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE-NEXT: ; ModuleID = {{.+}}
; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE: *** InstSimplifyPass ***
; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE-NEXT: ModuleID = {{.+}}

; CHECK-MULT-PASSES-FILTER-FUNC: *** Initial IR ***
; CHECK-MULT-PASSES-FILTER-FUNC-NEXT: define i32 @f()
; CHECK-MULT-PASSES-FILTER-FUNC: *** InstSimplifyPass ***
; CHECK-MULT-PASSES-FILTER-FUNC-NEXT: define i32 @f()
