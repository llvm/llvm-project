; RUN: llc -O3 < %s -aarch64-enable-atomic-cfg-tidy=0 -aarch64-enable-gep-opt=false -verify-machineinstrs | FileCheck %s
target triple = "arm64-apple-ios"

; rdar://12462006
; CSE between "icmp reg reg" and "sub reg reg".
; Both can be in the same basic block or in different basic blocks.
define i8* @t1(i8* %base, i32* nocapture %offset, i32 %size) nounwind {
entry:
; CHECK-LABEL: t1:
; CHECK: subs
; CHECK-NOT: cmp
; CHECK-NOT: sub
; CHECK: b.ge
; CHECK: sub
; CHECK: sub
; CHECK-NOT: sub
; CHECK: ret
 %0 = load i32, i32* %offset, align 4
 %cmp = icmp slt i32 %0, %size
 %s = sub nsw i32 %0, %size
 br i1 %cmp, label %return, label %if.end

if.end:
 %sub = sub nsw i32 %0, %size
 %s2 = sub nsw i32 %s, %size
 %s3 = sub nsw i32 %sub, %s2
 store i32 %s3, i32* %offset, align 4
 %add.ptr = getelementptr inbounds i8, i8* %base, i32 %sub
 br label %return

return:
 %retval.0 = phi i8* [ %add.ptr, %if.end ], [ null, %entry ]
 ret i8* %retval.0
}

; CSE between "icmp reg imm" and "sub reg imm".
define i8* @t2(i8* %base, i32* nocapture %offset) nounwind {
entry:
; CHECK-LABEL: t2:
; CHECK: subs
; CHECK-NOT: cmp
; CHECK-NOT: sub
; CHECK: b.lt
; CHECK-NOT: sub
; CHECK: ret
 %0 = load i32, i32* %offset, align 4
 %cmp = icmp slt i32 %0, 1
 br i1 %cmp, label %return, label %if.end

if.end:
 %sub = sub nsw i32 %0, 1
 store i32 %sub, i32* %offset, align 4
 %add.ptr = getelementptr inbounds i8, i8* %base, i32 %sub
 br label %return

return:
 %retval.0 = phi i8* [ %add.ptr, %if.end ], [ null, %entry ]
 ret i8* %retval.0
}
