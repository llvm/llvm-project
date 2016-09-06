; ; RUN: opt -analyze -scalar-evolution < %s | FileCheck %s

define void @infer.sext.0(i1* %c, i32 %start, i32* %buf) {
; CHECK-LABEL: Classifying expressions for: @infer.sext.0
 entry:
  br label %loop

 loop:
  %counter = phi i32 [ 0, %entry ], [ %counter.inc, %loop ]
  %idx = phi i32 [ %start, %entry ], [ %idx.inc, %loop ]
  %idx.inc = add nsw i32 %idx, 1
  %idx.inc.sext = sext i32 %idx.inc to i64
; CHECK: %idx.inc.sext = sext i32 %idx.inc to i64
; CHECK-NEXT: -->  {(1 + (sext i32 %start to i64))<nsw>,+,1}<nsw><%loop>

  %buf.gep = getelementptr inbounds i32, i32* %buf, i32 %idx.inc
  %val = load i32, i32* %buf.gep

  %condition = icmp eq i32 %counter, 1
  %counter.inc = add i32 %counter, 1
  br i1 %condition, label %exit, label %loop

 exit:
  ret void
}

define void @infer.zext.0(i1* %c, i32 %start, i32* %buf) {
; CHECK-LABEL: Classifying expressions for: @infer.zext.0
 entry:
  br label %loop

 loop:
  %counter = phi i32 [ 0, %entry ], [ %counter.inc, %loop ]
  %idx = phi i32 [ %start, %entry ], [ %idx.inc, %loop ]
  %idx.inc = add nuw i32 %idx, 1
  %idx.inc.sext = zext i32 %idx.inc to i64
; CHECK: %idx.inc.sext = zext i32 %idx.inc to i64
; CHECK-NEXT: -->  {(1 + (zext i32 %start to i64))<nuw><nsw>,+,1}<nuw><%loop>

  %buf.gep = getelementptr inbounds i32, i32* %buf, i32 %idx.inc
  %val = load i32, i32* %buf.gep

  %condition = icmp eq i32 %counter, 1
  %counter.inc = add i32 %counter, 1
  br i1 %condition, label %exit, label %loop

 exit:
  ret void
}

define void @infer.sext.1(i32 %start, i1* %c) {
; CHECK-LABEL: Classifying expressions for: @infer.sext.1
 entry:
  %start.mul = mul i32 %start, 4
  %start.real = add i32 %start.mul, 2
  br label %loop

 loop:
  %idx = phi i32 [ %start.real, %entry ], [ %idx.inc, %loop ]
  %idx.sext = sext i32 %idx to i64
; CHECK: %idx.sext = sext i32 %idx to i64
; CHECK-NEXT:  -->  {(2 + (sext i32 (4 * %start) to i64))<nsw>,+,2}<nsw><%loop>
  %idx.inc = add nsw i32 %idx, 2
  %condition = load i1, i1* %c
  br i1 %condition, label %exit, label %loop

 exit:
  ret void
}

define void @infer.sext.2(i1* %c, i8 %start) {
; CHECK-LABEL: Classifying expressions for: @infer.sext.2
 entry:
  %start.inc = add i8 %start, 1
  %entry.condition = icmp slt i8 %start, 127
  br i1 %entry.condition, label %loop, label %exit

 loop:
  %idx = phi i8 [ %start.inc, %entry ], [ %idx.inc, %loop ]
  %idx.sext = sext i8 %idx to i16
; CHECK: %idx.sext = sext i8 %idx to i16
; CHECK-NEXT: -->  {(1 + (sext i8 %start to i16))<nsw>,+,1}<nsw><%loop>
  %idx.inc = add nsw i8 %idx, 1
  %condition = load volatile i1, i1* %c
  br i1 %condition, label %exit, label %loop

 exit:
  ret void
}

define void @infer.zext.1(i1* %c, i8 %start) {
; CHECK-LABEL: Classifying expressions for: @infer.zext.1
 entry:
  %start.inc = add i8 %start, 1
  %entry.condition = icmp ult i8 %start, 255
  br i1 %entry.condition, label %loop, label %exit

 loop:
  %idx = phi i8 [ %start.inc, %entry ], [ %idx.inc, %loop ]
  %idx.zext = zext i8 %idx to i16
; CHECK: %idx.zext = zext i8 %idx to i16
; CHECK-NEXT: -->  {(1 + (zext i8 %start to i16))<nuw><nsw>,+,1}<nuw><%loop>
  %idx.inc = add nuw i8 %idx, 1
  %condition = load volatile i1, i1* %c
  br i1 %condition, label %exit, label %loop

 exit:
  ret void
}
