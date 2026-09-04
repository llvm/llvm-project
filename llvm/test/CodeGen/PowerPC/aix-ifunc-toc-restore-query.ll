; This testcase is for testing the positive return values of the
; TOCRestoreNeededForCallToImplementation query, specifically the type of
; functions that are considered DSO local on AIX.

; REQUIRES: asserts
; RUN: llc < %s -mtriple=powerpc64-ibm-aix-xcoff --function-sections  -filetype=obj -o /dev/null -debug-only=asmprinter 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=powerpc-ibm-aix-xcoff --function-sections  -filetype=obj -o /dev/null -debug-only=asmprinter 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64-ibm-aix-xcoff -ifunc-local-if-proven=1 -o /dev/null -debug-only=asmprinter 2>&1 | FileCheck %s

; CHECK: foo_ext_hidden_decl is dso_local
; CHECK: foo_ext_hidden_def is dso_local
; CHECK: foo_ext_hidden_weak_def is dso_local
; CHECK: foo_ext_protected_decl is dso_local
; CHECK: foo_ext_protected_def is dso_local
; CHECK: foo_ext_protected_weak_def is dso_local
; CHECK: foo_static is dso_local

; The following decls/defs are dso_local in the IR, and it matches the behaviour on AIX in practice
; (1) a hidden/protected declaration should have a matching definition in the same shared object,
;     with the matching visibility. So the definition is not interposable due to hidden/protected.
; (2) a hidden/protected definition is not interposable.
; (3) attribute weak has no effect on interposition, and if a strong definition in the same shared object
;     exists, then it's a user error to have that definition have conflicting visibility.
;     In practice, on AIX the linker will silently pick the strong definition and keep it's visibility
;     ignoring what's on the weak definition.
;     On Linux, both ld and lld pick the strong definition but give it the most restrictive visibility based
;     on the candidates available (so a weak hidden and a strong default will yield a hidden strong)
;
declare hidden void @foo_ext_hidden_decl(...)                 ; (1)

declare protected void @foo_ext_protected_decl(...)           ; (1)

define hidden void @foo_ext_hidden_def() {                    ; (2)
entry:
  ret void
}

define protected void @foo_ext_protected_def() {              ; (2)
entry:
  ret void
}

define weak hidden void @foo_ext_hidden_weak_def() {          ; (3)
entry:
  ret void
}

define weak protected void @foo_ext_protected_weak_def() {    ; (3)
entry:
  ret void
}

define internal void @foo_static() {
entry:
  ret void
}

@foo = ifunc void (...), ptr @resolve_foo

@x = global i32 0, align 4

@switch.table.bar = private unnamed_addr constant [6 x ptr] [ptr @foo_ext_hidden_decl, ptr @foo_ext_hidden_def, ptr @foo_ext_hidden_weak_def, ptr @foo_ext_protected_decl, ptr @foo_ext_protected_def, ptr @foo_ext_protected_weak_def], align 4

define internal nonnull ptr @resolve_foo() {
entry:
  %x = load i32, ptr @x, align 4
  %0 = icmp ult i32 %x, 6
  br i1 %0, label %switch.lookup, label %return

switch.lookup:                                    ; preds = %entry
  %switch.gep = getelementptr inbounds nuw ptr, ptr @switch.table.bar, i32 %x
  %switch.load = load ptr, ptr %switch.gep, align 4
  br label %return

return:                                           ; preds = %entry, %switch.lookup
  %retval.0 = phi ptr [ %switch.load, %switch.lookup ], [ @foo_static, %entry ]
  ret ptr %retval.0
}
