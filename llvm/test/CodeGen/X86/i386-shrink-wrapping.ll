; RUN: llc %s -o - -enable-shrink-wrap=true -no-x86-call-frame-opt | FileCheck %s --check-prefix=CHECK --check-prefix=ENABLE
; RUN: llc %s -o - -enable-shrink-wrap=false -no-x86-call-frame-opt | FileCheck %s --check-prefix=CHECK --check-prefix=DISABLE
target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-apple-macosx10.5"

@a = common global i32 0, align 4
@d = internal unnamed_addr global i1 false
@b = common global i32 0, align 4
@e = common global i8 0, align 1
@f = common global i8 0, align 1
@c = common global i32 0, align 4
@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1


; Check that we are clobbering the flags when they are live-in of the
; prologue block and the prologue needs to adjust the stack.
; PR25607.
;
; CHECK-LABEL: eflagsLiveInPrologue:
;
; DISABLE: pushl
; DISABLE-NEXT: subl $8, %esp
;
; CHECK: movl L_a$non_lazy_ptr, [[A:%[a-z]+]]
; CHECK-NEXT: cmpl $0, ([[A]])
; CHECK-NEXT: je [[PREHEADER_LABEL:LBB[0-9_]+]]
;
; CHECK: movb $1, _d
;
; CHECK: [[PREHEADER_LABEL]]:
; CHECK-NEXT: movl L_b$non_lazy_ptr, [[B:%[a-z]+]]
; CHECK-NEXT: movl ([[B]]), [[TMP1:%[a-z]+]]
; CHECK-NEXT: testl [[TMP1]], [[TMP1]]
; CHECK-NEXT: je  [[FOREND_LABEL:LBB[0-9_]+]]
;
; Skip the loop.
; [...]
;
; The for.end block is split to accomadate the different selects.
; We are interested in the one with the call, so skip until the branch.
; CHECK: [[FOREND_LABEL]]:

; ENABLE: pushl
; ENABLE-NEXT: subl $8, %esp

; CHECK: xorl [[CMOVE_VAL:%edx]], [[CMOVE_VAL]]
; CHECK-NEXT: cmpb $0, _d
; CHECK-NEXT: movl $6, [[IMM_VAL:%ecx]]
; The eflags is used in the next instruction.
; If that instruction disappear, we are not exercising the bug
; anymore.
; CHECK-NEXT: cmovnel [[CMOVE_VAL]], [[IMM_VAL]]

; CHECK-NEXT: L_e$non_lazy_ptr, [[E:%[a-z]+]]
; CHECK-NEXT: movb %cl, ([[E]])
; CHECK-NEXT: leal 1(%ecx), %esi

; CHECK: calll _varfunc
; Set the return value to 0.
; CHECK-NEXT: xorl %eax, %eax
; CHECK-NEXT: addl $8, %esp
; CHECK-NEXT: popl
; CHECK-NEXT: retl
define i32 @eflagsLiveInPrologue() #0 {
entry:
  %tmp = load i32, i32* @a, align 4
  %tobool = icmp eq i32 %tmp, 0
  br i1 %tobool, label %for.cond.preheader, label %if.then

if.then:                                          ; preds = %entry
  store i1 true, i1* @d, align 1
  br label %for.cond.preheader

for.cond.preheader:                               ; preds = %if.then, %entry
  %tmp1 = load i32, i32* @b, align 4
  %tobool14 = icmp eq i32 %tmp1, 0
  br i1 %tobool14, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %for.cond.preheader
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  br label %for.body

for.end:                                          ; preds = %for.cond.preheader
  %.b3 = load i1, i1* @d, align 1
  %tmp2 = select i1 %.b3, i8 0, i8 6
  store i8 %tmp2, i8* @e, align 1
  %tmp3 = load i8, i8* @e, align 1
  %conv = sext i8 %tmp3 to i32
  %add = add nsw i32 %conv, 1
  %rem = srem i32 %tmp1, %add
  store i32 %rem, i32* @c, align 4
  %conv2 = select i1 %.b3, i32 0, i32 6
  %call = tail call i32 (i8*, ...) @varfunc(i8* nonnull getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i32 %conv2) #1
  ret i32 0
}

; Function Attrs: nounwind
declare i32 @varfunc(i8* nocapture readonly, ...) #0

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-features"="+mmx,+sse" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
