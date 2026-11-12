; RUN: opt -S -passes=licm -o - %s | FileCheck %s
;
; Be sure that we don't hoist loads incorrectly if a loop has conditional UB.
; See PR36228.

declare void @check(i8)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i32, i1)

; CHECK-LABEL: define void @buggy
define void @buggy(ptr %src, ptr %kOne) {
entry:
  %dst = alloca [1 x i8], align 1
  store i8 42, ptr %dst, align 1
  %srcval = load i16, ptr %src
  br label %while.cond

while.cond:                                       ; preds = %if.end, %entry
  %dp.0 = phi ptr [ %dst, %entry ], [ %dp.1, %if.end ]
  %0 = load volatile i1, ptr %kOne, align 4
  br i1 %0, label %if.else, label %if.then

if.then:                                          ; preds = %while.cond
  store i8 9, ptr %dp.0, align 1
  br label %if.end

if.else:                                          ; preds = %while.cond
  call void @llvm.memcpy.p0.p0.i64(ptr %dp.0, ptr %src, i64 2, i32 1, i1 false)
  %dp.new = getelementptr inbounds i8, ptr %dp.0, i64 1
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %dp.1 = phi ptr [ %dp.0, %if.then ], [ %dp.new, %if.else ]
  ; CHECK: %1 = load i8, ptr %dst
  %1 = load i8, ptr %dst, align 1
  ; CHECK-NEXT: call void @check(i8 %1)
  call void @check(i8 %1)
  br label %while.cond
}
