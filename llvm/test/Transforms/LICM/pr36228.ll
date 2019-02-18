; RUN: opt -S -licm -o - %s | FileCheck %s
;
; Be sure that we don't hoist loads incorrectly if a loop has conditional UB.
; See PR36228.

declare void @check(i8)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i32, i1)

; CHECK-LABEL: define void @buggy
define void @buggy(i8* %src, i1* %kOne) {
entry:
  %dst = alloca [1 x i8], align 1
  %0 = getelementptr inbounds [1 x i8], [1 x i8]* %dst, i64 0, i64 0
  store i8 42, i8* %0, align 1
  %src16 = bitcast i8* %src to i16*
  %srcval = load i16, i16* %src16
  br label %while.cond

while.cond:                                       ; preds = %if.end, %entry
  %dp.0 = phi i8* [ %0, %entry ], [ %dp.1, %if.end ]
  %1 = load volatile i1, i1* %kOne, align 4
  br i1 %1, label %if.else, label %if.then

if.then:                                          ; preds = %while.cond
  store i8 9, i8* %dp.0, align 1
  br label %if.end

if.else:                                          ; preds = %while.cond
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dp.0, i8* %src, i64 2, i32 1, i1 false)
  %dp.new = getelementptr inbounds i8, i8* %dp.0, i64 1
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %dp.1 = phi i8* [ %dp.0, %if.then ], [ %dp.new, %if.else ]
  ; CHECK: %2 = load i8, i8* %0
  %2 = load i8, i8* %0, align 1
  ; CHECK-NEXT: call void @check(i8 %2)
  call void @check(i8 %2)
  br label %while.cond
}
