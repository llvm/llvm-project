; RUN: opt -S -jump-threading -dce < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @test1(i32 %a, i32 %b) #0 {
entry:
  %cmp = icmp sgt i32 %a, 5
  tail call void @llvm.assume(i1 %cmp)
  %cmp1 = icmp sgt i32 %b, 1234
  br i1 %cmp1, label %if.then, label %if.else

; CHECK-LABEL: @test1
; CHECK: icmp sgt i32 %a, 5
; CHECK: call void @llvm.assume
; CHECK-NOT: icmp sgt i32 %a, 3
; CHECK: ret i32

if.then:                                          ; preds = %entry
  %cmp2 = icmp sgt i32 %a, 3
  br i1 %cmp2, label %if.then3, label %return

if.then3:                                         ; preds = %if.then
  tail call void (...) @bar() #1
  br label %return

if.else:                                          ; preds = %entry
  tail call void (...) @car() #1
  br label %return

return:                                           ; preds = %if.else, %if.then, %if.then3
  %retval.0 = phi i32 [ 1, %if.then3 ], [ 0, %if.then ], [ 0, %if.else ]
  ret i32 %retval.0
}

define i32 @test2(i32 %a) #0 {
entry:
  %cmp = icmp sgt i32 %a, 5
  tail call void @llvm.assume(i1 %cmp)
  %cmp1 = icmp sgt i32 %a, 3
  br i1 %cmp1, label %if.then, label %return

; CHECK-LABEL: @test2
; CHECK: icmp sgt i32 %a, 5
; CHECK: tail call void @llvm.assume
; CHECK: tail call void (...) @bar()
; CHECK: ret i32 1


if.then:                                          ; preds = %entry
  tail call void (...) @bar() #1
  br label %return

return:                                           ; preds = %entry, %if.then
  %retval.0 = phi i32 [ 1, %if.then ], [ 0, %entry ]
  ret i32 %retval.0
}

@g = external global i32

; Check that we do prove a fact using an assume within the block.
; FIXME: We can fold the assume based on the semantics of assume.
; CHECK-LABEL: @can_fold_assume
; CHECK: %notnull = icmp ne i32* %array, null
; CHECK-NEXT: call void @llvm.assume(i1 %notnull)
; CHECK-NEXT: ret void
define void @can_fold_assume(i32* %array) {
  %notnull = icmp ne i32* %array, null
  call void @llvm.assume(i1 %notnull)
  br i1 %notnull, label %normal, label %error

normal:
  ret void

error:
  store atomic i32 0, i32* @g unordered, align 4
  ret void
}

declare void @f(i1)
declare void @exit()
; We can fold the assume but not the uses before the assume.
define void @dont_fold_incorrectly(i32* %array) {
; CHECK-LABEL:@dont_fold_incorrectly
; CHECK: @f(i1 %notnull)
; CHECK-NEXT: exit()
; CHECK-NEXT: assume(i1 %notnull)
; CHECK-NEXT: ret void
  %notnull = icmp ne i32* %array, null
  call void @f(i1 %notnull)
  call void @exit()
  call void @llvm.assume(i1 %notnull)
  br i1 %notnull, label %normal, label %error

normal:
  ret void

error:
  store atomic i32 0, i32* @g unordered, align 4
  ret void
}

; Function Attrs: nounwind
declare void @llvm.assume(i1) #1

declare void @bar(...)

declare void @car(...)

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind }

