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
; CHECK:  %[[ADD]] = add i32 %[[INDEX]], 1
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
; CHECK:  %[[TOPTR:[0-9A-Za-z.]+]] = inttoptr i32 %[[ADD:[0-9A-Za-z.]+]] to i32*
; CHECK:  %[[INDEX:[0-9A-Za-z.]+]] = phi i32 [ %[[ADD:[0-9A-Za-z.]+]], %bb ], [ %Offset, %entry ]
; CHECK:  %[[ADD]] = add i32 %[[INDEX]], 1
; CHECK:  %cond = icmp sgt i32 %[[INDEX]], 100
; CHECK:  br i1 %cond, label %bb2, label %bb
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
