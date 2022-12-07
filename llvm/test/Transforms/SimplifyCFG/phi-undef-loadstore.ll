; RUN: opt -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S < %s | FileCheck %s

declare void @bar() nounwind

define i32 @test1(i32* %a, i32 %b, i32* %c, i32 %d) nounwind {
entry:
  %tobool = icmp eq i32 %b, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  tail call void @bar() nounwind
  br label %if.end7

if.else:                                          ; preds = %entry
  %tobool3 = icmp eq i32 %d, 0
  br i1 %tobool3, label %if.end7, label %if.then4

if.then4:                                         ; preds = %if.else
  tail call void @bar() nounwind
  br label %if.end7

if.end7:                                          ; preds = %if.else, %if.then4, %if.then
  %x.0 = phi i32* [ %a, %if.then ], [ %c, %if.then4 ], [ null, %if.else ]
  %tmp9 = load i32, i32* %x.0
  ret i32 %tmp9

; CHECK-LABEL: @test1(
; CHECK: if.else:
; CHECK: br label %if.end7

; CHECK: phi i32* [ %a, %if.then ], [ %c, %if.else ]
}

define i32 @test1_no_null_opt(i32* %a, i32 %b, i32* %c, i32 %d) nounwind #0 {
entry:
  %tobool = icmp eq i32 %b, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  tail call void @bar() nounwind
  br label %if.end7

if.else:                                          ; preds = %entry
  %tobool3 = icmp eq i32 %d, 0
  br i1 %tobool3, label %if.end7, label %if.then4

if.then4:                                         ; preds = %if.else
  tail call void @bar() nounwind
  br label %if.end7

if.end7:                                          ; preds = %if.else, %if.then4, %if.then
  %x.0 = phi i32* [ %a, %if.then ], [ %c, %if.then4 ], [ null, %if.else ]
  %tmp9 = load i32, i32* %x.0
  ret i32 %tmp9

; CHECK-LABEL: @test1_no_null_opt(
; CHECK: if.then:
; CHECK: if.else:
; CHECK: if.then4:
; CHECK: br label %if.end7
; CHECK: if.end7:
; CHECK-NEXT: phi i32* [ %a, %if.then ], [ %c, %if.then4 ], [ null, %if.else ]
}

define i32 @test2(i32* %a, i32 %b, i32* %c, i32 %d) nounwind {
entry:
  %tobool = icmp eq i32 %b, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  tail call void @bar() nounwind
  br label %if.end7

if.else:                                          ; preds = %entry
  %tobool3 = icmp eq i32 %d, 0
  br i1 %tobool3, label %if.end7, label %if.then4

if.then4:                                         ; preds = %if.else
  tail call void @bar() nounwind
  br label %if.end7

if.end7:                                          ; preds = %if.else, %if.then4, %if.then
  %x.0 = phi i32* [ %a, %if.then ], [ null, %if.then4 ], [ null, %if.else ]
  %tmp9 = load i32, i32* %x.0
  ret i32 %tmp9
; CHECK-LABEL: @test2(
; CHECK: if.else:
; CHECK: unreachable

; CHECK-NOT: phi
}

define i32 @test2_no_null_opt(i32* %a, i32 %b, i32* %c, i32 %d) nounwind #0 {
entry:
  %tobool = icmp eq i32 %b, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  tail call void @bar() nounwind
  br label %if.end7

if.else:                                          ; preds = %entry
  %tobool3 = icmp eq i32 %d, 0
  br i1 %tobool3, label %if.end7, label %if.then4

if.then4:                                         ; preds = %if.else
  tail call void @bar() nounwind
  br label %if.end7

if.end7:                                          ; preds = %if.else, %if.then4, %if.then
  %x.0 = phi i32* [ %a, %if.then ], [ null, %if.then4 ], [ null, %if.else ]
  %tmp9 = load i32, i32* %x.0
  ret i32 %tmp9
; CHECK-LABEL: @test2_no_null_opt(
; CHECK: if.then:
; CHECK: if.else:
; CHECK: if.then4:
; CHECK: if.end7:
; CHECK-NEXT: phi i32* [ %a, %if.then ], [ null, %if.then4 ], [ null, %if.else ]
}

define i32 @test3(i32* %a, i32 %b, i32* %c, i32 %d) nounwind {
entry:
  %tobool = icmp eq i32 %b, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  tail call void @bar() nounwind
  br label %if.end7

if.else:                                          ; preds = %entry
  %tobool3 = icmp eq i32 %d, 0
  br i1 %tobool3, label %if.end7, label %if.then4

if.then4:                                         ; preds = %if.else
  tail call void @bar() nounwind
  br label %if.end7

if.end7:                                          ; preds = %if.else, %if.then4, %if.then
  %x.0 = phi i32* [ %a, %if.then ], [ null, %if.then4 ], [ null, %if.else ]
  tail call void @bar() nounwind
  %tmp9 = load i32, i32* %x.0
  ret i32 %tmp9
; CHECK-LABEL: @test3(
; CHECK: if.end7:
; CHECK: phi i32* [ %a, %if.then ], [ null, %if.then4 ], [ null, %if.else ]
}

define i32 @test3_no_null_opt(i32* %a, i32 %b, i32* %c, i32 %d) nounwind #0 {
entry:
  %tobool = icmp eq i32 %b, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  tail call void @bar() nounwind
  br label %if.end7

if.else:                                          ; preds = %entry
  %tobool3 = icmp eq i32 %d, 0
  br i1 %tobool3, label %if.end7, label %if.then4

if.then4:                                         ; preds = %if.else
  tail call void @bar() nounwind
  br label %if.end7

if.end7:                                          ; preds = %if.else, %if.then4, %if.then
  %x.0 = phi i32* [ %a, %if.then ], [ null, %if.then4 ], [ null, %if.else ]
  tail call void @bar() nounwind
  %tmp9 = load i32, i32* %x.0
  ret i32 %tmp9
; CHECK-LABEL: @test3_no_null_opt(
; CHECK: if.then:
; CHECK: if.else:
; CHECK: if.then4:
; CHECK: if.end7:
; CHECK-NEXT: phi i32* [ %a, %if.then ], [ null, %if.then4 ], [ null, %if.else ]
}

define i32 @test4(i32* %a, i32 %b, i32* %c, i32 %d) nounwind {
entry:
  %tobool = icmp eq i32 %b, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  tail call void @bar() nounwind
  br label %if.end7

if.else:                                          ; preds = %entry
  %tobool3 = icmp eq i32 %d, 0
  br i1 %tobool3, label %if.end7, label %if.then4

if.then4:                                         ; preds = %if.else
  tail call void @bar() nounwind
  br label %if.end7

if.end7:                                          ; preds = %if.else, %if.then4, %if.then
  %x.0 = phi i32* [ %a, %if.then ], [ null, %if.then4 ], [ null, %if.else ]
  %gep = getelementptr i32, i32* %x.0, i32 10
  %tmp9 = load i32, i32* %gep
  %tmp10 = or i32 %tmp9, 1
  store i32 %tmp10, i32* %gep
  ret i32 %tmp9
; CHECK-LABEL: @test4(
; CHECK-NOT: phi
}

define i32 @test4_no_null_opt(i32* %a, i32 %b, i32* %c, i32 %d) nounwind #0 {
entry:
  %tobool = icmp eq i32 %b, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  tail call void @bar() nounwind
  br label %if.end7

if.else:                                          ; preds = %entry
  %tobool3 = icmp eq i32 %d, 0
  br i1 %tobool3, label %if.end7, label %if.then4

if.then4:                                         ; preds = %if.else
  tail call void @bar() nounwind
  br label %if.end7

if.end7:                                          ; preds = %if.else, %if.then4, %if.then
  %x.0 = phi i32* [ %a, %if.then ], [ null, %if.then4 ], [ null, %if.else ]
  %gep = getelementptr i32, i32* %x.0, i32 10
  %tmp9 = load i32, i32* %gep
  %tmp10 = or i32 %tmp9, 1
  store i32 %tmp10, i32* %gep
  ret i32 %tmp9
; CHECK-LABEL: @test4_no_null_opt(
; CHECK: if.then:
; CHECK: if.else:
; CHECK: if.then4:
; CHECK: if.end7:
; CHECK-NEXT: phi i32* [ %a, %if.then ], [ null, %if.then4 ], [ null, %if.else ]
}

attributes #0 = { null_pointer_is_valid }
