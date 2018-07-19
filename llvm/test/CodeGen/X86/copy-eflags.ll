; RUN: llc -o - %s | FileCheck %s
; This tests for the problem originally reported in http://llvm.org/PR25951
target triple = "i686-unknown-linux-gnu"

@b = common global i8 0, align 1
@c = common global i32 0, align 4
@a = common global i8 0, align 1
@d = common global i8 0, align 1
@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1

; CHECK-LABEL: func:
; This tests whether eax is properly saved/restored around the
; lahf/sahf instruction sequences. We make mem op volatile to prevent
; their reordering to avoid spills.


define i32 @func() {
entry:
  %bval = load i8, i8* @b
  %inc = add i8 %bval, 1
  store volatile i8 %inc, i8* @b
  %cval = load volatile i32, i32* @c
  %inc1 = add nsw i32 %cval, 1
  store volatile i32 %inc1, i32* @c
  %aval = load volatile i8, i8* @a
  %inc2 = add i8 %aval, 1
  store volatile i8 %inc2, i8* @a
; Copy flags produced by the incb of %inc1 to a register, need to save+restore
; eax around it. The flags will be reused by %tobool.
; CHECK: pushl %eax
; CHECK: seto %al
; CHECK: lahf
; CHECK: movl %eax, [[REG:%[a-z]+]]
; CHECK: popl %eax
  %cmp = icmp eq i8 %aval, %bval
  %conv5 = zext i1 %cmp to i8
  store i8 %conv5, i8* @d
  %tobool = icmp eq i32 %inc1, 0
; We restore flags with an 'addb, sahf' sequence, need to save+restore eax
; around it.
; CHECK: pushl %eax
; CHECK: movl [[REG]], %eax
; CHECK: addb $127, %al
; CHECK: sahf
; CHECK: popl %eax
  br i1 %tobool, label %if.end, label %if.then

if.then:
  %conv6 = sext i8 %inc to i32
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i32 %conv6)
  br label %if.end

if.end:
  ret i32 0
}

declare i32 @printf(i8* nocapture readonly, ...)
