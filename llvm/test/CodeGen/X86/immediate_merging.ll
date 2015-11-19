; RUN: llc -o - -mtriple=i386-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -o - -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

@a = common global i32 0, align 4
@b = common global i32 0, align 4
@c = common global i32 0, align 4
@e = common global i32 0, align 4
@x = common global i32 0, align 4
@f = common global i32 0, align 4
@h = common global i32 0, align 4
@i = common global i32 0, align 4

; Test -Os to make sure immediates with multiple users don't get pulled in to
; instructions.
define i32 @foo() optsize {
; CHECK-LABEL: foo:
; CHECK: movl $1234, [[R1:%[a-z]+]]
; CHECK-NOT: movl $1234, a
; CHECK-NOT: movl $1234, b
; CHECK-NOT: movl $12, c
; CHECK-NOT: cmpl $12, e
; CHECK: movl [[R1]], a
; CHECK: movl [[R1]], b

entry:
  store i32 1234, i32* @a
  store i32 1234, i32* @b
  store i32 12, i32* @c
  %0 = load i32, i32* @e
  %cmp = icmp eq i32 %0, 12
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 1, i32* @x
  br label %if.end

; New block.. Make sure 1234 isn't live across basic blocks from before.
; CHECK: movl $1234, f
; CHECK: movl $555, [[R3:%[a-z]+]]
; CHECK-NOT: movl $555, h
; CHECK-NOT: addl $555, i
; CHECK: movl [[R3]], h
; CHECK: addl [[R3]], i

if.end:                                           ; preds = %if.then, %entry
  store i32 1234, i32* @f
  store i32 555, i32* @h
  %1 = load i32, i32* @i
  %add1 = add nsw i32 %1, 555
  store i32 %add1, i32* @i
  ret i32 0
}

; Test -O2 to make sure that all immediates get pulled in to their users.
define i32 @foo2() {
; CHECK-LABEL: foo2:
; CHECK: movl $1234, a
; CHECK: movl $1234, b

entry:
  store i32 1234, i32* @a
  store i32 1234, i32* @b

  ret i32 0
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) #1

@AA = common global [100 x i8] zeroinitializer, align 1

; memset gets lowered in DAG. Constant merging should hoist all the
; immediates used to store to the individual memory locations. Make
; sure we don't directly store the immediates.
define void @foomemset() optsize {
; CHECK-LABEL: foomemset:
; CHECK-NOT: movl ${{.*}}, AA
; CHECK: mov{{l|q}} %{{e|r}}ax, AA

entry:
  call void @llvm.memset.p0i8.i32(i8* getelementptr inbounds ([100 x i8], [100 x i8]* @AA, i32 0, i32 0), i8 33, i32 24, i32 1, i1 false)
  ret void
}
