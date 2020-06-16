; RUN: opt -instcombine -S  < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:64"

define i32 *@test1(i32* %A, i32 %Offset) {
entry:
  %tmp = getelementptr inbounds i32, i32* %A, i32 %Offset
  br label %bb

bb:
  %RHS = phi i32* [ %RHS.next, %bb ], [ %tmp, %entry ]
  %LHS = getelementptr inbounds i32, i32* %A, i32 100
  %RHS.next = getelementptr inbounds i32, i32* %RHS, i64 1
  %cond = icmp ult i32 * %LHS, %RHS
  br i1 %cond, label %bb2, label %bb

bb2:
  ret i32* %RHS

; CHECK-LABEL: @test1(
; CHECK:  %[[INDEX:[0-9A-Za-z.]+]] = phi i32 [ %[[ADD:[0-9A-Za-z.]+]], %bb ], [ %Offset, %entry ]
; CHECK:  %[[ADD]] = add nsw i32 %[[INDEX]], 1
; CHECK:  %cond = icmp sgt i32 %[[INDEX]], 100
; CHECK:  br i1 %cond, label %bb2, label %bb
; CHECK:  %[[PTR:[0-9A-Za-z.]+]] = getelementptr inbounds i32, i32* %A, i32 %[[INDEX]]
; CHECK:  ret i32* %[[PTR]]
}

define i32 *@test2(i32 %A, i32 %Offset) {
entry:
  %A.ptr = inttoptr i32 %A to i32*
  %tmp = getelementptr inbounds i32, i32* %A.ptr, i32 %Offset
  br label %bb

bb:
  %RHS = phi i32* [ %RHS.next, %bb ], [ %tmp, %entry ]
  %LHS = getelementptr inbounds i32, i32* %A.ptr, i32 100
  %RHS.next = getelementptr inbounds i32, i32* %RHS, i64 1
  %cmp0 = ptrtoint i32 *%LHS to i32
  %cmp1 = ptrtoint i32 *%RHS to i32
  %cond = icmp ult i32 %cmp0, %cmp1
  br i1 %cond, label %bb2, label %bb

bb2:
  ret i32* %RHS

; CHECK-LABEL: @test2(
; CHECK:  %[[INDEX:[0-9A-Za-z.]+]] = phi i32 [ %[[ADD:[0-9A-Za-z.]+]], %bb ], [ %Offset, %entry ]
; CHECK:  %[[ADD]] = add nsw i32 %[[INDEX]], 1
; CHECK:  %cond = icmp sgt i32 %[[INDEX]], 100
; CHECK:  br i1 %cond, label %bb2, label %bb
; CHECK:  %[[TOPTR:[0-9A-Za-z.]+]] = inttoptr i32 %[[ADD:[0-9A-Za-z.]+]] to i32*
; CHECK:  %[[PTR:[0-9A-Za-z.]+]] = getelementptr inbounds i32, i32* %[[TOPTR]], i32 %[[INDEX]]
; CHECK:  ret i32* %[[PTR]]
}

; Perform the transformation only if we know that the GEPs used are inbounds.
define i32 *@test3(i32* %A, i32 %Offset) {
entry:
  %tmp = getelementptr i32, i32* %A, i32 %Offset
  br label %bb

bb:
  %RHS = phi i32* [ %RHS.next, %bb ], [ %tmp, %entry ]
  %LHS = getelementptr i32, i32* %A, i32 100
  %RHS.next = getelementptr i32, i32* %RHS, i64 1
  %cond = icmp ult i32 * %LHS, %RHS
  br i1 %cond, label %bb2, label %bb

bb2:
  ret i32* %RHS

; CHECK-LABEL: @test3(
; CHECK-NOT:  %cond = icmp sgt i32 %{{[0-9A-Za-z.]+}}, 100
}

; An inttoptr that requires an extension or truncation will be opaque when determining
; the base pointer. In this case we can still perform the transformation by considering
; A.ptr as being the base pointer.
define i32 *@test4(i16 %A, i32 %Offset) {
entry:
  %A.ptr = inttoptr i16 %A to i32*
  %tmp = getelementptr inbounds i32, i32* %A.ptr, i32 %Offset
  br label %bb

bb:
  %RHS = phi i32* [ %RHS.next, %bb ], [ %tmp, %entry ]
  %LHS = getelementptr inbounds i32, i32* %A.ptr, i32 100
  %RHS.next = getelementptr inbounds i32, i32* %RHS, i64 1
  %cmp0 = ptrtoint i32 *%LHS to i32
  %cmp1 = ptrtoint i32 *%RHS to i32
  %cond = icmp ult i32 %cmp0, %cmp1
  br i1 %cond, label %bb2, label %bb

bb2:
  ret i32* %RHS

; CHECK-LABEL: @test4(
; CHECK:  %cond = icmp sgt i32 %{{[0-9A-Za-z.]+}}, 100
}

declare i32* @fun_ptr()

define i32 *@test5(i32 %Offset) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
 %A = invoke i32 *@fun_ptr() to label %cont unwind label %lpad

cont:
  %tmp = getelementptr inbounds i32, i32* %A, i32 %Offset
  br label %bb

bb:
  %RHS = phi i32* [ %RHS.next, %bb ], [ %tmp, %cont ]
  %LHS = getelementptr inbounds i32, i32* %A, i32 100
  %RHS.next = getelementptr inbounds i32, i32* %RHS, i64 1
  %cond = icmp ult i32 * %LHS, %RHS
  br i1 %cond, label %bb2, label %bb

bb2:
  ret i32* %RHS

lpad:
  %l = landingpad { i8*, i32 } cleanup
  ret i32* null

; CHECK-LABEL: @test5(
; CHECK:  %[[INDEX:[0-9A-Za-z.]+]] = phi i32 [ %[[ADD:[0-9A-Za-z.]+]], %bb ], [ %Offset, %cont ]
; CHECK:  %[[ADD]] = add nsw i32 %[[INDEX]], 1
; CHECK:  %cond = icmp sgt i32 %[[INDEX]], 100
; CHECK:  br i1 %cond, label %bb2, label %bb
; CHECK:  %[[PTR:[0-9A-Za-z.]+]] = getelementptr inbounds i32, i32* %A, i32 %[[INDEX]]
; CHECK:  ret i32* %[[PTR]]
}

declare i32 @fun_i32()

define i32 *@test6(i32 %Offset) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
 %A = invoke i32 @fun_i32() to label %cont unwind label %lpad

cont:
  %A.ptr = inttoptr i32 %A to i32*
  %tmp = getelementptr inbounds i32, i32* %A.ptr, i32 %Offset
  br label %bb

bb:
  %RHS = phi i32* [ %RHS.next, %bb ], [ %tmp, %cont ]
  %LHS = getelementptr inbounds i32, i32* %A.ptr, i32 100
  %RHS.next = getelementptr inbounds i32, i32* %RHS, i64 1
  %cond = icmp ult i32 * %LHS, %RHS
  br i1 %cond, label %bb2, label %bb

bb2:
  ret i32* %RHS

lpad:
  %l = landingpad { i8*, i32 } cleanup
  ret i32* null

; CHECK-LABEL: @test6(
; CHECK:  %[[INDEX:[0-9A-Za-z.]+]] = phi i32 [ %[[ADD:[0-9A-Za-z.]+]], %bb ], [ %Offset, %cont ]
; CHECK:  %[[ADD]] = add nsw i32 %[[INDEX]], 1
; CHECK:  %cond = icmp sgt i32 %[[INDEX]], 100
; CHECK:  br i1 %cond, label %bb2, label %bb
; CHECK:  %[[TOPTR:[0-9A-Za-z.]+]] = inttoptr i32 %[[ADD:[0-9A-Za-z.]+]] to i32*
; CHECK:  %[[PTR:[0-9A-Za-z.]+]] = getelementptr inbounds i32, i32* %[[TOPTR]], i32 %[[INDEX]]
; CHECK:  ret i32* %[[PTR]]
}


@pr30402 = constant i64 3
define i1 @test7() {
entry:
  br label %bb7

bb7:                                              ; preds = %bb10, %entry-block
  %phi = phi i64* [ @pr30402, %entry ], [ getelementptr inbounds (i64, i64* @pr30402, i32 1), %bb7 ]
  %cmp = icmp eq i64* %phi, getelementptr inbounds (i64, i64* @pr30402, i32 1)
  br i1 %cmp, label %bb10, label %bb7

bb10:
  ret i1 %cmp
}
; CHECK-LABEL: @test7(
; CHECK:  %[[phi:.*]] = phi i64* [ @pr30402, %entry ], [ getelementptr inbounds (i64, i64* @pr30402, i32 1), %bb7 ]
; CHECK:  %[[cmp:.*]] = icmp eq i64* %[[phi]], getelementptr inbounds (i64, i64* @pr30402, i32 1)
; CHECK: ret i1 %[[cmp]]


declare i32 @__gxx_personality_v0(...)

define i1 @test8(i64* %in, i64 %offset) {
entry:

 %ld = load i64, i64* %in, align 8
 %casti8 = inttoptr i64 %ld to i8*
 %gepi8 = getelementptr inbounds i8, i8* %casti8, i64 %offset
 %cast = bitcast i8* %gepi8 to i32**
 %ptrcast = inttoptr i64 %ld to i32**
 %gepi32 = getelementptr inbounds i32*, i32** %ptrcast, i64 1
 %cmp = icmp eq i32** %gepi32, %cast
 ret i1 %cmp


; CHECK-LABEL: @test8(
; CHECK-NOT: icmp eq i32 %{{[0-9A-Za-z.]+}}, 1
}
