; This testcase is for testing the positive return values of the
; TOCRestoreNeededForCallToImplementation query, specifically the type of
; functions that are considered DSO local on AIX.

; RUN: llc < %s -mtriple=powerpc64-ibm-aix-xcoff --function-sections  -filetype=obj -o /dev/null
; RUN: llc < %s -mtriple=powerpc-ibm-aix-xcoff --function-sections  -filetype=obj -o /dev/null
; RUN: llc < %s -mtriple=powerpc64-ibm-aix-xcoff -ifunc-local-if-proven=1 -o /dev/null

@x = global i32 0, align 4

; "hidden" or "internal" visibility function. "internal" would also have the
; local_unnamed_addr IR attribute but we don't test it here because it's
; considered by the query we're testing.
define hidden i32 @foo_hidden() #0 {
entry:
  ret i32 2
}

define protected i32 @foo_protected() #0 {
entry:
  ret i32 3
}

define internal i32 @foo_static() #0 {
entry:
  ret i32 1
}

@foo = ifunc i32 (...), ptr @resolve_foo

define internal nonnull ptr @resolve_foo() {
entry:
  %0 = load i32, ptr @x, align 4
  %switch.selectcmp = icmp eq i32 %0, 1
  %switch.select = select i1 %switch.selectcmp, ptr @foo_hidden, ptr @foo_static
  %switch.selectcmp3 = icmp eq i32 %0, 2
  %switch.select4 = select i1 %switch.selectcmp3, ptr @foo_protected, ptr %switch.select
  ret ptr %switch.select4
}
