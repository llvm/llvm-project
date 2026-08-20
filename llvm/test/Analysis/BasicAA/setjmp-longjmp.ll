; Test that BasicAA conservatively returns MayAlias for local allocas in
; functions containing returns_twice calls (e.g. setjmp/sigsetjmp), to prevent
; miscompilation via longjmp re-entry paths invisible in the forward CFG.
;
; Reproduces: https://github.com/llvm/llvm-project/issues/198967
;
; Without the fix, GVN incorrectly concludes that the store through %p_val
; (which may alias %i via a longjmp re-entry) cannot modify %i, then DSE
; eliminates the "store i32 13, ptr %i" as dead. The fix adds an O(1)
; check for the contains_returns_twice_call function attribute at the point
; where isNotCapturedBefore would otherwise return true.

; RUN: opt < %s -aa-pipeline=basic-aa -passes=gvn,dse -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x  = external global i32
@ii = external global i32
@p  = external global ptr

declare void @redo()
declare void @checkpoint() #0

; Control case: function has the same CFG structure but no contains_returns_twice_call
; attribute. GVN/DSE can legally eliminate "store i32 13, ptr %i" because in the
; forward CFG the only path to if.else does not pass through if.then (where &i is
; stored to @p). BasicAA correctly returns NoAlias for (%i, %p_val) here.
;
; Fix case CHECK directives are listed here so that CHECK-NOT is correctly scoped
; between the two CHECK-LABELs: it checks that the store is absent in bar_no_attr
; and present in bar_with_attr.
;
; CHECK-LABEL: define void @bar_no_attr(
; CHECK-NOT:   store i32 13, ptr %i
; CHECK-LABEL: define void @bar_with_attr(
; CHECK:       store i32 13, ptr %i
define void @bar_no_attr() {
entry:
  %i = alloca i32, align 4
  %x_val = load i32, ptr @x
  %cond = icmp ne i32 %x_val, 0
  br i1 %cond, label %if.then, label %if.else

if.then:
  store ptr %i, ptr @p
  call void @redo()
  br label %if.end

if.else:
  ; Bug: without the attribute, GVN sees "store i32 42, ptr %p_val" as
  ; NoAlias with %i (the capture in if.then is not visible in the forward
  ; CFG), so it forwards 13 through the load and DSE removes this store.
  store i32 13, ptr %i
  %p_val = load ptr, ptr @p
  store i32 42, ptr %p_val
  %ii_val = load i32, ptr %i
  store i32 %ii_val, ptr @ii
  br label %if.end

if.end:
  ret void
}

; Fix case: same structure with contains_returns_twice_call on the function
; (set automatically by Clang when emitting setjmp / sigsetjmp / any
; returns_twice call). BasicAA conservatively returns MayAlias for (%i,
; %p_val) because a longjmp can re-enter at the checkpoint() site with @p
; already holding &i, making the if.then capture visible to if.else.
;
; The "store i32 13, ptr %i" MUST be preserved; removing it is a
; miscompilation. (CHECKs for this function are listed above to correctly
; scope the CHECK-NOT between the two CHECK-LABELs.)
define void @bar_with_attr() #1 {
entry:
  %i = alloca i32, align 4
  call void @checkpoint() #0
  %x_val = load i32, ptr @x
  %cond = icmp ne i32 %x_val, 0
  br i1 %cond, label %if.then, label %if.else

if.then:
  ; On the first pass: stores &i into @p, then calls redo().
  ; redo() performs longjmp(buf) which re-enters at checkpoint() above.
  ; On re-entry: if.then runs again setting p = &i, then if.else runs
  ; with @p == &i, making "store i32 42, ptr %p_val" alias "store i32 13, ptr %i".
  store ptr %i, ptr @p
  call void @redo()
  br label %if.end

if.else:
  store i32 13, ptr %i
  %p_val = load ptr, ptr @p
  store i32 42, ptr %p_val   ; may write to %i via longjmp re-entry path
  %ii_val = load i32, ptr %i
  store i32 %ii_val, ptr @ii
  br label %if.end

if.end:
  ret void
}

attributes #0 = { returns_twice }
attributes #1 = { contains_returns_twice_call }
