; RUN: opt < %s -indvars -S "-default-data-layout=e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64" | FileCheck %s
; RUN: opt < %s -indvars -S "-default-data-layout=e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32" | FileCheck %s
;
; PR11279: Assertion !IVLimit->getType()->isPointerTy()
;
; Test LinearFunctionTestReplace of a pointer-type loop counter. Note
; that BECount may or may not be a pointer type. A pointer type
; BECount doesn't really make sense, but that's what falls out of
; SCEV. Since it's an i8*, it has unit stride so we never adjust the
; SCEV expression in a way that would convert it to an integer type.

; CHECK: @testnullptrptr
; CHECK: loop:
; CHECK: icmp ne
define i8 @testnullptrptr(i8* %buf, i8* %end) nounwind {
  br label %loopguard

loopguard:
  %guard = icmp ult i8* null, %end
  br i1 %guard, label %preheader, label %exit

preheader:
  br label %loop

loop:
  %p.01.us.us = phi i8* [ null, %preheader ], [ %gep, %loop ]
  %s = phi i8 [0, %preheader], [%snext, %loop]
  %gep = getelementptr inbounds i8* %p.01.us.us, i64 1
  %snext = load i8* %gep
  %cmp = icmp ult i8* %gep, %end
  br i1 %cmp, label %loop, label %exit

exit:
  ret i8 %snext
}

; CHECK: @testptrptr
; CHECK: loop:
; CHECK: icmp ne
define i8 @testptrptr(i8* %buf, i8* %end) nounwind {
  br label %loopguard

loopguard:
  %guard = icmp ult i8* %buf, %end
  br i1 %guard, label %preheader, label %exit

preheader:
  br label %loop

loop:
  %p.01.us.us = phi i8* [ %buf, %preheader ], [ %gep, %loop ]
  %s = phi i8 [0, %preheader], [%snext, %loop]
  %gep = getelementptr inbounds i8* %p.01.us.us, i64 1
  %snext = load i8* %gep
  %cmp = icmp ult i8* %gep, %end
  br i1 %cmp, label %loop, label %exit

exit:
  ret i8 %snext
}

; CHECK: @testnullptrint
; CHECK: loop:
; CHECK: icmp ne
define i8 @testnullptrint(i8* %buf, i8* %end) nounwind {
  br label %loopguard

loopguard:
  %bi = ptrtoint i8* %buf to i32
  %ei = ptrtoint i8* %end to i32
  %cnt = sub i32 %ei, %bi
  %guard = icmp ult i32 0, %cnt
  br i1 %guard, label %preheader, label %exit

preheader:
  br label %loop

loop:
  %p.01.us.us = phi i8* [ null, %preheader ], [ %gep, %loop ]
  %iv = phi i32 [ 0, %preheader ], [ %ivnext, %loop ]
  %s = phi i8 [0, %preheader], [%snext, %loop]
  %gep = getelementptr inbounds i8* %p.01.us.us, i64 1
  %snext = load i8* %gep
  %ivnext = add i32 %iv, 1
  %cmp = icmp ult i32 %ivnext, %cnt
  br i1 %cmp, label %loop, label %exit

exit:
  ret i8 %snext
}

; CHECK: @testptrint
; CHECK: loop:
; CHECK: icmp ne
define i8 @testptrint(i8* %buf, i8* %end) nounwind {
  br label %loopguard

loopguard:
  %bi = ptrtoint i8* %buf to i32
  %ei = ptrtoint i8* %end to i32
  %cnt = sub i32 %ei, %bi
  %guard = icmp ult i32 %bi, %cnt
  br i1 %guard, label %preheader, label %exit

preheader:
  br label %loop

loop:
  %p.01.us.us = phi i8* [ %buf, %preheader ], [ %gep, %loop ]
  %iv = phi i32 [ %bi, %preheader ], [ %ivnext, %loop ]
  %s = phi i8 [0, %preheader], [%snext, %loop]
  %gep = getelementptr inbounds i8* %p.01.us.us, i64 1
  %snext = load i8* %gep
  %ivnext = add i32 %iv, 1
  %cmp = icmp ult i32 %ivnext, %cnt
  br i1 %cmp, label %loop, label %exit

exit:
  ret i8 %snext
}

; IV and BECount have two different pointer types here.
define void @testnullptr([512 x i8]* %base) nounwind {
entry:
  %add.ptr1603 = getelementptr [512 x i8]* %base, i64 0, i64 512
  br label %preheader

preheader:
  %cmp1604192 = icmp ult i8* undef, %add.ptr1603
  br i1 %cmp1604192, label %for.body, label %for.end1609

for.body:
  %r.17193 = phi i8* [ %incdec.ptr1608, %for.body ], [ null, %preheader ]
  %incdec.ptr1608 = getelementptr i8* %r.17193, i64 1
  %cmp1604 = icmp ult i8* %incdec.ptr1608, %add.ptr1603
  br i1 %cmp1604, label %for.body, label %for.end1609

for.end1609:
  unreachable
}
