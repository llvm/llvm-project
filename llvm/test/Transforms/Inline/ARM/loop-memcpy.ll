; RUN: opt -passes=inline %s -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7m-arm-none-eabi"

; CHECK-LABEL: define void @matcpy
define void @matcpy(ptr %dest, ptr %source, i32 %num) #0 {
entry:
  %0 = ptrtoint ptr %dest to i32
  %1 = ptrtoint ptr %source to i32
  %2 = xor i32 %0, %1
  %3 = and i32 %2, 3
  %cmp = icmp eq i32 %3, 0
  br i1 %cmp, label %if.then, label %if.else20

if.then:                                          ; preds = %entry
  %sub = sub i32 0, %0
  %and2 = and i32 %sub, 3
  %add = or i32 %and2, 4
  %cmp3 = icmp ugt i32 %add, %num
  br i1 %cmp3, label %if.else, label %if.then4

if.then4:                                         ; preds = %if.then
  %sub5 = sub i32 %num, %and2
  %shr = and i32 %sub5, -4
  %sub7 = sub i32 %sub5, %shr
  %tobool = icmp eq i32 %and2, 0
  br i1 %tobool, label %if.end, label %if.then8

if.then8:                                         ; preds = %if.then4
; CHECK: call fastcc void @memcpy
  call fastcc void @memcpy(ptr %dest, ptr %source, i32 %and2) #0
  %add.ptr = getelementptr inbounds i8, ptr %dest, i32 %and2
  %add.ptr9 = getelementptr inbounds i8, ptr %source, i32 %and2
  br label %if.end

if.end:                                           ; preds = %if.then4, %if.then8
  %p_dest.0 = phi ptr [ %add.ptr, %if.then8 ], [ %dest, %if.then4 ]
  %p_source.0 = phi ptr [ %add.ptr9, %if.then8 ], [ %source, %if.then4 ]
  %tobool14 = icmp eq i32 %sub7, 0
  br i1 %tobool14, label %if.end22, label %if.then15

if.then15:                                        ; preds = %if.end
  %add.ptr13 = getelementptr inbounds i8, ptr %p_source.0, i32 %shr
  %add.ptr11 = getelementptr inbounds i8, ptr %p_dest.0, i32 %shr
; CHECK: call fastcc void @memcpy
  call fastcc void @memcpy(ptr %add.ptr11, ptr %add.ptr13, i32 %sub7) #0
  br label %if.end22

if.else:                                          ; preds = %if.then
  call fastcc void @memcpy(ptr %dest, ptr %source, i32 %num) #0
  br label %if.end22

if.else20:                                        ; preds = %entry
  call fastcc void @memcpy(ptr %dest, ptr %source, i32 %num) #0
  br label %if.end22

if.end22:                                         ; preds = %if.then15, %if.end, %if.else, %if.else20
  ret void
}

; CHECK-LABEL: define internal void @memcpy
define internal void @memcpy(ptr nocapture %dest, ptr nocapture readonly %source, i32 %num) #0 {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %num.addr.0 = phi i32 [ %num, %entry ], [ %dec, %while.body ]
  %p_dest.0 = phi ptr [ %dest, %entry ], [ %incdec.ptr1, %while.body ]
  %p_source.0 = phi ptr [ %source, %entry ], [ %incdec.ptr, %while.body ]
  %cmp = icmp eq i32 %num.addr.0, 0
  br i1 %cmp, label %while.end, label %while.body

while.body:                                       ; preds = %while.cond
  %incdec.ptr = getelementptr inbounds i8, ptr %p_source.0, i32 1
  %0 = load i8, ptr %p_source.0, align 1
  %incdec.ptr1 = getelementptr inbounds i8, ptr %p_dest.0, i32 1
  store i8 %0, ptr %p_dest.0, align 1
  %dec = add i32 %num.addr.0, -1
  br label %while.cond

while.end:                                        ; preds = %while.cond
  ret void
}

attributes #0 = { minsize optsize }

