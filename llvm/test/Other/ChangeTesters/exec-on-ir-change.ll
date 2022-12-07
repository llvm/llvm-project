; Simple checks of -exec-on-ir-change=cat functionality
;
; Simple functionality check.
; RUN: opt -S -exec-on-ir-change=cat -passes=instsimplify 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-SIMPLE
;
; Check that only the passes that change the IR are printed and that the
; others (including g) are filtered out.
; RUN: opt -S -exec-on-ir-change=cat -passes=instsimplify -filter-print-funcs=f  2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FUNC-FILTER
;
; Check that the reporting of IRs respects -print-module-scope
; RUN: opt -S -exec-on-ir-change=cat -passes=instsimplify -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-PRINT-MOD-SCOPE
;
; Check that the reporting of IRs respects -print-module-scope
; RUN: opt -S -exec-on-ir-change=cat -passes=instsimplify -filter-print-funcs=f -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FUNC-FILTER-MOD-SCOPE
;
; Check that reporting of multiple functions happens
; RUN: opt -S -exec-on-ir-change=cat -passes=instsimplify -filter-print-funcs="f,g" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FILTER-MULT-FUNC
;
; Check that the reporting of IRs respects -filter-passes
; RUN: opt -S -exec-on-ir-change=cat -passes="instsimplify,no-op-function" -filter-passes="no-op-function" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FILTER-PASSES
;
; Check that the reporting of IRs respects -filter-passes with multiple passes
; RUN: opt -S -exec-on-ir-change=cat -passes="instsimplify,no-op-function" -filter-passes="no-op-function,instsimplify" 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FILTER-MULT-PASSES
;
; Check that the reporting of IRs respects both -filter-passes and -filter-print-funcs
; RUN: opt -S -exec-on-ir-change=cat -passes="instsimplify,no-op-function" -filter-passes="no-op-function,instsimplify" -filter-print-funcs=f 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FILTER-FUNC-PASSES
;
; Check that the reporting of IRs respects -filter-passes, -filter-print-funcs and -print-module-scope
; RUN: opt -S -exec-on-ir-change=cat -passes="instsimplify,no-op-function" -filter-passes="no-op-function,instsimplify" -filter-print-funcs=f -print-module-scope 2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-FILTER-FUNC-PASSES-MOD-SCOPE
;
; Check that repeated passes that change the IR are printed and that the
; others (including g) are filtered out.  Note that the second time
; instsimplify is run on f, it does not change the IR
; RUN: opt -S -exec-on-ir-change=cat -passes="instsimplify,instsimplify" -filter-print-funcs=f  2>&1 -o /dev/null < %s | FileCheck %s --check-prefix=CHECK-MULT-PASSES-FILTER-FUNC
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

; CHECK-SIMPLE: ; ModuleID = {{.+}}
; CHECK-SIMPLE: cat:{{.*}}Initial IR
; CHECK-SIMPLE: define i32 @g()
; CHECK-SIMPLE: cat:{{.*}}InstSimplifyPass
; CHECK-SIMPLE: define i32 @f()
; CHECK-SIMPLE: cat:{{.*}}InstSimplifyPass

; CHECK-FUNC-FILTER: define i32 @f()
; CHECK-FUNC-FILTER: cat:{{.*}}Initial IR
; CHECK-FUNC-FILTER: define i32 @f()
; CHECK-FUNC-FILTER: cat:{{.*}}InstSimplifyPass

; CHECK-PRINT-MOD-SCOPE: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE: cat:{{.*}}Initial IR
; CHECK-PRINT-MOD-SCOPE: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE: cat:{{.*}}InstSimplifyPass
; CHECK-PRINT-MOD-SCOPE: ModuleID = {{.+}}
; CHECK-PRINT-MOD-SCOPE: cat:{{.*}}InstSimplifyPass

; CHECK-FUNC-FILTER-MOD-SCOPE: ; ModuleID = {{.+}}
; CHECK-FUNC-FILTER-MOD-SCOPE: cat:{{.*}}Initial IR
; CHECK-FUNC-FILTER-MOD-SCOPE: ModuleID = {{.+}}
; CHECK-FUNC-FILTER-MOD-SCOPE: cat:{{.*}}InstSimplifyPass

; CHECK-FILTER-MULT-FUNC: define i32 @g()
; CHECK-FILTER-MULT-FUNC: cat:{{.*}}Initial IR
; CHECK-FILTER-MULT-FUNC: define i32 @g()
; CHECK-FILTER-MULT-FUNC: cat:{{.*}}InstSimplifyPass
; CHECK-FILTER-MULT-FUNC: define i32 @f()
; CHECK-FILTER-MULT-FUNC: cat:{{.*}}InstSimplifyPass

; CHECK-FILTER-PASSES: define i32 @g()
; CHECK-FILTER-PASSES: cat:{{.*}}Initial IR

; CHECK-FILTER-MULT-PASSES: define i32 @g()
; CHECK-FILTER-MULT-PASSES: cat:{{.*}}Initial IR
; CHECK-FILTER-MULT-PASSES: define i32 @g()
; CHECK-FILTER-MULT-PASSES: cat:{{.*}}InstSimplifyPass
; CHECK-FILTER-MULT-PASSES: define i32 @f()
; CHECK-FILTER-MULT-PASSES: cat:{{.*}}InstSimplifyPass

; CHECK-FILTER-FUNC-PASSES: define i32 @f()
; CHECK-FILTER-FUNC-PASSES: cat:{{.*}}Initial IR
; CHECK-FILTER-FUNC-PASSES: define i32 @f()
; CHECK-FILTER-FUNC-PASSES: cat:{{.*}}InstSimplifyPass

; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE: ; ModuleID = {{.+}}
; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE: cat:{{.*}}Initial IR
; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE: ModuleID = {{.+}}
; CHECK-FILTER-FUNC-PASSES-MOD-SCOPE: cat:{{.*}}InstSimplifyPass

; CHECK-MULT-PASSES-FILTER-FUNC: define i32 @f()
; CHECK-MULT-PASSES-FILTER-FUNC: cat:{{.*}}Initial IR
; CHECK-MULT-PASSES-FILTER-FUNC: define i32 @f()
; CHECK-MULT-PASSES-FILTER-FUNC: cat:{{.*}}InstSimplifyPass
