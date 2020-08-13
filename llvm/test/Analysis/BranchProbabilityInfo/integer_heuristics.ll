; RUN: opt < %s -analyze -branch-prob | FileCheck %s

@A = global i32 0, align 4
@B = global i32 0, align 4

; CHECK-LABEL: eq_opaque_minus_one
define void @eq_opaque_minus_one(i32* %base) {
entry:
  %const = bitcast i32 -1 to i32
  %tmp1 = load i32, i32* @B, align 4
  br label %for.body

; CHECK: edge for.body -> if.then probability is 0x30000000 / 0x80000000 = 37.50%
; CHECK: edge for.body -> for.inc probability is 0x50000000 / 0x80000000 = 62.50%
for.body:
  %tmp4 = phi i32 [ %tmp1, %entry ], [ %tmp7, %for.inc ]
  %inc.iv = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %storemerge176.in = getelementptr inbounds i32, i32* %base, i32 %inc.iv
  %storemerge176 = load i32, i32* %storemerge176.in, align 4
  store i32 %storemerge176, i32* @A, align 4
  %cmp20 = icmp eq i32 %storemerge176, %const
  br i1 %cmp20, label %if.then, label %for.inc

if.then:
  %lnot.ext = zext i1 %cmp20 to i32
  store i32 %lnot.ext, i32* @B, align 4
  br label %for.inc

for.inc:
  %tmp7 = phi i32 [ %tmp4, %for.body ], [ %lnot.ext, %if.then ]
  %inc = add nuw nsw i32 %inc.iv, 1
  %cmp9 = icmp ult i32 %inc, 401
  br i1 %cmp9, label %for.body, label %exit

exit:
  ret void
}

; CHECK-LABEL: ne_opaque_minus_one
define void @ne_opaque_minus_one(i32* %base) {
entry:
  %const = bitcast i32 -1 to i32
  %tmp1 = load i32, i32* @B, align 4
  br label %for.body

; CHECK: edge for.body -> if.then probability is 0x50000000 / 0x80000000 = 62.50%
; CHECK: edge for.body -> for.inc probability is 0x30000000 / 0x80000000 = 37.50%
for.body:
  %tmp4 = phi i32 [ %tmp1, %entry ], [ %tmp7, %for.inc ]
  %inc.iv = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %storemerge176.in = getelementptr inbounds i32, i32* %base, i32 %inc.iv
  %storemerge176 = load i32, i32* %storemerge176.in, align 4
  store i32 %storemerge176, i32* @A, align 4
  %cmp20 = icmp ne i32 %storemerge176, %const
  br i1 %cmp20, label %if.then, label %for.inc

if.then:
  %lnot.ext = zext i1 %cmp20 to i32
  store i32 %lnot.ext, i32* @B, align 4
  br label %for.inc

for.inc:
  %tmp7 = phi i32 [ %tmp4, %for.body ], [ %lnot.ext, %if.then ]
  %inc = add nuw nsw i32 %inc.iv, 1
  %cmp9 = icmp ult i32 %inc, 401
  br i1 %cmp9, label %for.body, label %exit

exit:
  ret void
}

; CHECK-LABEL: sgt_opaque_minus_one
define void @sgt_opaque_minus_one(i32* %base) {
entry:
  %const = bitcast i32 -1 to i32
  %tmp1 = load i32, i32* @B, align 4
  br label %for.body

; CHECK: edge for.body -> if.then probability is 0x50000000 / 0x80000000 = 62.50%
; CHECK: edge for.body -> for.inc probability is 0x30000000 / 0x80000000 = 37.50%
for.body:
  %tmp4 = phi i32 [ %tmp1, %entry ], [ %tmp7, %for.inc ]
  %inc.iv = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %storemerge176.in = getelementptr inbounds i32, i32* %base, i32 %inc.iv
  %storemerge176 = load i32, i32* %storemerge176.in, align 4
  store i32 %storemerge176, i32* @A, align 4
  %cmp20 = icmp sgt i32 %storemerge176, %const
  br i1 %cmp20, label %if.then, label %for.inc

if.then:
  %lnot.ext = zext i1 %cmp20 to i32
  store i32 %lnot.ext, i32* @B, align 4
  br label %for.inc

for.inc:
  %tmp7 = phi i32 [ %tmp4, %for.body ], [ %lnot.ext, %if.then ]
  %inc = add nuw nsw i32 %inc.iv, 1
  %cmp9 = icmp ult i32 %inc, 401
  br i1 %cmp9, label %for.body, label %exit

exit:
  ret void
}

declare void @foo() 

; CHECK-LABEL: foo1
define i32 @foo1(i32 %x, i32 %y, i8 signext %z, i8 signext %w) {
entry: 
  %c = icmp eq i32 %x, %y
  br i1 %c, label %then, label %else
; CHECK: edge entry -> then probability is 0x30000000 / 0x80000000 = 37.50%
; CHECK: edge entry -> else probability is 0x50000000 / 0x80000000 = 62.50%
then:
  tail call void @foo()
  br label %else
; CHECK: edge then -> else probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
else:
  %v = phi i8 [ %z, %then ], [ %w, %entry ]
  %r = sext i8 %v to i32
  ret i32 %r
}

; CHECK-LABEL: foo2
define i32 @foo2(i32 %x, i32 %y, i8 signext %z, i8 signext %w) {
entry: 
  %c = icmp ne i32 %x, %y
  br i1 %c, label %then, label %else
; CHECK: edge entry -> then probability is 0x50000000 / 0x80000000 = 62.50%
; CHECK: edge entry -> else probability is 0x30000000 / 0x80000000 = 37.50%
then:
  br label %else
; CHECK: edge then -> else probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
else:
  %v = phi i8 [ %z, %then ], [ %w, %entry ]
  %r = sext i8 %v to i32
  ret i32 %r
}

; CHECK-LABEL: foo3
define i32 @foo3(i32 %x, i32 %y, i8 signext %z, i8 signext %w) {
entry: 
  %c = icmp ult i32 %x, %y
  br i1 %c, label %then, label %else
; CHECK: edge entry -> then probability is 0x50000000 / 0x80000000 = 62.50%
; CHECK: edge entry -> else probability is 0x30000000 / 0x80000000 = 37.50%
then:
  br label %else
; CHECK: edge then -> else probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]
else:
  %v = phi i8 [ %z, %then ], [ %w, %entry ]
  %r = sext i8 %v to i32
  ret i32 %r
}
