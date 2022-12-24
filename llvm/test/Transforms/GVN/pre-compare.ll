; RUN: opt -passes=gvn -S < %s | FileCheck %s

; C source:
;
;   void f(int x) {
;     if (x != 1)
;       puts (x == 2 ? "a" : "b");
;     for (;;) {
;       puts("step 1");
;       if (x == 2)
;         continue;
;       printf("step 2: %d\n", x);
;     }
;   }
;
; If we PRE %cmp3, CodeGenPrepare won't be able to sink the compare down to its
; uses, and we are forced to keep both %x and %cmp3 in registers in the loop.
;
; It is just as cheap to recompute the icmp against %x as it is to compare a
; GPR against 0. On x86-64, the br i1 %cmp3 becomes:
;
;   testb %r12b, %r12b
;   jne	LBB0_3
;
; The sunk icmp is:
;
;   cmpl $2, %ebx
;   je	LBB0_3
;
; This is just as good, and it doesn't require a separate register.
;
; CHECK-NOT: phi i1

@.str = private unnamed_addr constant [2 x i8] c"a\00", align 1
@.str1 = private unnamed_addr constant [2 x i8] c"b\00", align 1
@.str2 = private unnamed_addr constant [7 x i8] c"step 1\00", align 1
@.str3 = private unnamed_addr constant [12 x i8] c"step 2: %d\0A\00", align 1

define void @f(i32 %x) noreturn nounwind uwtable ssp {
entry:
  %cmp = icmp eq i32 %x, 1
  br i1 %cmp, label %for.cond.preheader, label %if.then

if.then:                                          ; preds = %entry
  %cmp1 = icmp eq i32 %x, 2
  %cond = select i1 %cmp1, ptr @.str, ptr @.str1
  %call = tail call i32 @puts(ptr %cond) nounwind
  br label %for.cond.preheader

for.cond.preheader:                               ; preds = %entry, %if.then
  %cmp3 = icmp eq i32 %x, 2
  br label %for.cond

for.cond:                                         ; preds = %for.cond.backedge, %for.cond.preheader
  %call2 = tail call i32 @puts(ptr @.str2) nounwind
  br i1 %cmp3, label %for.cond.backedge, label %if.end5

if.end5:                                          ; preds = %for.cond
  %call6 = tail call i32 (ptr, ...) @printf(ptr @.str3, i32 %x) nounwind
  br label %for.cond.backedge

for.cond.backedge:                                ; preds = %if.end5, %for.cond
  br label %for.cond
}

declare i32 @puts(ptr nocapture) nounwind

declare i32 @printf(ptr nocapture, ...) nounwind
