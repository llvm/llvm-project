; RUN: llc -march=mipsel -mcpu=mips32r6 -relocation-model=static \
; RUN:     -disable-mips-delay-filler < %s | FileCheck %s -check-prefix=STATIC32
; RUN: llc -march=mipsel -mcpu=mips64r6 -relocation-model=pic -target-abi n64 \
; RUN:     -disable-mips-delay-filler < %s | FileCheck %s -check-prefix=PIC

; Function Attrs: nounwind
define void @l()  {
entry:
; PIC: jalrc $25
  %call = tail call i32 @k()
; PIC: jalrc $25
  %call1 = tail call i32 @j()
  %cmp = icmp eq i32 %call, %call1
; CHECK: bnec
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry:
; STATIC: nop
; STATIC: jal
; PIC: jalrc $25
  tail call void @f(i32 signext -2)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
; CHECK: jrc  $ra
  ret void
}

declare i32 @k()

declare i32 @j()

declare void @f(i32 signext) 

; Function Attrs: define void @l2()  {
define void @l2()  {
entry:
; PIC: jalrc $25
  %call = tail call i32 @k()
; PIC: jalrc $25
  %call1 = tail call i32 @i()
  %cmp = icmp eq i32 %call, %call1
; CHECK: beqc
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry:
; STATIC: nop
; STATIC: jal
; PIC: jalrc $25
  tail call void @f(i32 signext -1)
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
; CHECK: jrc  $ra
  ret void
}

declare i32 @i()

; Function Attrs: nounwind
define void @l3()  {
entry:
; PIC: jalrc $25
  %call = tail call i32 @k()
  %cmp = icmp slt i32 %call, 0
; CHECK: bgez
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry:
; STATIC: nop
; STATIC: jal
; PIC: jalrc $25
  tail call void @f(i32 signext 0)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
; CHECK: jrc $ra
  ret void
}

; Function Attrs: nounwind
define void @l4()  {
entry:
  %call = tail call i32 @k()
  %cmp = icmp slt i32 %call, 1
; CHECK: bgtzc
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry:
; STATIC: nop
; STATIC: jal
  tail call void @f(i32 signext 1)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
; CHECK: jrc $ra
  ret void
}

; Function Attrs: nounwind
define void @l5()  {
entry:
; PIC: jalrc $25
  %call = tail call i32 @k()
; PIC: jalrc $25
  %cmp = icmp sgt i32 %call, 0
; CHECK: blezc
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry:
; STATIC: nop
; STATIC: jal
; PIC: jalrc $25
  tail call void @f(i32 signext 2) 
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
; CHECK: jrc  $ra
  ret void
}

; Function Attrs: nounwind
define void @l6()  {
entry:
; PIC: jalrc $25
  %call = tail call i32 @k()
; PIC: jalrc $25
  %cmp = icmp sgt i32 %call, -1
; CHECK: bltzc
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry:
; STATIC: nop
; STATIC: jal
; PIC: jalrc $25
  tail call void @f(i32 signext 3)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
; CHECK: jrc $ra
  ret void
}

; Function Attrs: nounwind
define void @l7()  {
entry:
; PIC: jalrc $25
  %call = tail call i32 @k()
  %cmp = icmp eq i32 %call, 0
; CHECK: bnezc
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry:
; STATIC: nop
; STATIC: jal
; PIC: jalrc $25
  tail call void @f(i32 signext 4)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
; CHECK: jrc  $ra
  ret void
}

; Function Attrs: nounwind
define void @l8()  {
entry:
; PIC: jalrc $25
  %call = tail call i32 @k()
  %cmp = icmp eq i32 %call, 0
; CHECK: beqzc
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry:
; STATIC: nop
; STATIC: jal
; PIC: jalrc $25
  tail call void @f(i32 signext 5)
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
; CHECK: jrc  $ra
  ret void
}

define i32 @l9(i8* ()* %i) #0 {
entry:
  %i.addr = alloca i8* ()*, align 4
  store i8* ()* %i, i8* ()** %i.addr, align 4
; STATIC32: jal
; STATIC32: nop
; PIC: jalrc $25
  %call = call i32 @k()
; PIC: jalrc $25
  %cmp = icmp ne i32 %call, 0
; CHECK: beqzc
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %0 = load i8* ()*, i8* ()** %i.addr, align 4
; CHECK: jalrc $25
  %call1 = call i8* %0()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
; CHECK: jrc $ra
  ret i32 -1
}
