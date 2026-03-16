; RUN: opt -enable-scalar-pre=false -enable-pre -passes=gvn -S < %s | FileCheck %s --check-prefixes=CHECK

define void @kernel(ptr %arr, i8 %cond) {
entry:
  %tobool.not = icmp eq i8 %cond, 0
  %tmp7.pre = load i32, ptr %arr, align 4
  br i1 %tobool.not, label %if.end, label %if.then

; CHECK: if.then:
; CHECK-NEXT: [[ADD:%.*]] = add nsw i32 [[LOAD:%.*]], 2

if.then:                                          ; preds = %entry
  %add = add nsw i32 %tmp7.pre, 2
  %getElem = getelementptr inbounds nuw i8, ptr %arr, i64 8
  store i32 %add, ptr %getElem, align 4
  br label %if.end

; CHECK: if.end:
; CHECK-NEXT: [[ADD2:%.*]] = add nsw i32 [[LOAD:%.*]], 2

if.end:                                           ; preds = %if.then, %entry
  %add8 = add nsw i32 %tmp7.pre, 2
  %getElem1 = getelementptr inbounds nuw i8, ptr %arr, i64 12
  store i32 %add8, ptr %getElem1, align 4
  ret void
}