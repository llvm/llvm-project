; RUN: llc -relocation-model=pic -march=mipsel -mcpu=mips64r6 \
; RUN:     -disable-mips-delay-filler -target-abi=n64 < %s | FileCheck %s

; Function Attrs: nounwind
define void @l() {
entry:
; CHECK-LABEL:  l:
; CHECK: jalrc $25
  %call = tail call i64 @k()
; CHECK: jalrc $25
  %call1 = tail call i64 @j()
  %cmp = icmp eq i64 %call, %call1
; CHECK: bnec
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry:
; CHECK: jalrc $25
  tail call void @f(i64 signext -2)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
; CHECK: jrc  $ra
  ret void
}

declare i64 @k()

declare i64 @j()

declare void @f(i64 signext)

; Function Attrs: define void @l2() {
define void @l2() {
entry:
; CHECK-LABEL: l2:
; CHECK: jalrc $25
  %call = tail call i64 @k()
; CHECK: jalrc $25
  %call1 = tail call i64 @i()
  %cmp = icmp eq i64 %call, %call1
; CHECK: beqc
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry:
; CHECK: jalrc $25
  tail call void @f(i64 signext -1)
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
; CHECK: jrc  $ra
  ret void
}

declare i64 @i()

; Function Attrs: nounwind
define void @l3() {
entry:
; CHECK-LABEL: l3:
; CHECK: jalrc $25
  %call = tail call i64 @k()
  %cmp = icmp slt i64 %call, 0
; CHECK: bgez
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry:
; CHECK: jalrc $25
  tail call void @f(i64 signext 0)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
; CHECK: jrc $ra
  ret void
}

; Function Attrs: nounwind
define void @l4() {
entry:
; CHECK-LABEL: l4:
; CHECK: jalrc $25
  %call = tail call i64 @k()
  %cmp = icmp slt i64 %call, 1
; CHECK: bgtzc
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry:
  tail call void @f(i64 signext 1)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
; CHECK: jrc $ra
  ret void
}

; Function Attrs: nounwind
define void @l5() {
entry:
; CHECK-LABEL: l5:
; CHECK: jalrc $25
  %call = tail call i64 @k()
  %cmp = icmp sgt i64 %call, 0
; CHECK: blezc
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry:
; CHECK: jalrc $25
  tail call void @f(i64 signext 2)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
; CHECK: jrc  $ra
  ret void
}

; Function Attrs: nounwind
define void @l6() {
entry:
; CHECK-LABEL: l6:
; CHECK: jalrc $25
  %call = tail call i64 @k()
  %cmp = icmp sgt i64 %call, -1
; CHECK: bltzc
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry:
; CHECK: jalrc $25
  tail call void @f(i64 signext 3)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
; CHECK: jrc $ra
  ret void
}

; Function Attrs: nounwind
define void @l7() {
entry:
; CHECK-LABEL: l7:
; CHECK: jalrc $25
  %call = tail call i64 @k()
  %cmp = icmp eq i64 %call, 0
; CHECK: bnezc
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry:
; CHECK: jalrc $25
  tail call void @f(i64 signext 4)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
; CHECK: jrc  $ra
  ret void
}

; Function Attrs: nounwind
define void @l8() {
entry:
; CHECK-LABEL: l8:
; CHECK: jalrc $25
  %call = tail call i64 @k()
  %cmp = icmp eq i64 %call, 0
; CHECK: beqzc
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry:
; CHECK: jalrc $25
  tail call void @f(i64 signext 5)
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
; CHECK: jrc  $ra
  ret void
}

define i64 @l9(ptr %i) {
entry:
; CHECK-LABEL: l9:
  %i.addr = alloca ptr, align 4
  store ptr %i, ptr %i.addr, align 4
; CHECK: jalrc $25
  %call = call i64 @k()
  %cmp = icmp ne i64 %call, 0
; CHECK: beqzc
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %0 = load ptr, ptr %i.addr, align 4
; CHECK: jalrc $25
  %call1 = call ptr %0()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
; CHECK: jrc $ra
  ret i64 -1
}
