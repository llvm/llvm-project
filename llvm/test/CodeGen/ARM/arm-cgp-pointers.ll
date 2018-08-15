; RUN: llc -mtriple=thumbv8 -arm-disable-cgp=false %s -o - | FileCheck %s
; RUN: llc -mtriple=armv8 -arm-disable-cgp=false %s -o - | FileCheck %s

; CHECK-LABEL: phi_pointers
; CHECK-NOT: uxt
define void @phi_pointers(i16* %a, i16* %b, i8 zeroext %M, i8 zeroext %N) {
entry:
  %add = add nuw i8 %M, 1
  %and = and i8 %add, 1
  %cmp = icmp ugt i8 %add, %N
  %base = select i1 %cmp, i16* %a, i16* %b
  %other = select i1 %cmp, i16* %b, i16* %b
  br label %loop

loop:
  %ptr = phi i16* [ %base, %entry ], [ %gep, %loop ]
  %idx = phi i8 [ %and, %entry ], [ %inc, %loop ]
  %load = load i16, i16* %ptr, align 2
  %inc = add nuw nsw i8 %idx, 1
  %gep = getelementptr inbounds i16, i16* %ptr, i8 %inc
  %cond = icmp eq i16* %gep, %other
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}

; CHECK-LABEL: phi_pointers_null
; CHECK-NOT: uxt
define void @phi_pointers_null(i16* %a, i16* %b, i8 zeroext %M, i8 zeroext %N) {
entry:
  %add = add nuw i8 %M, 1
  %and = and i8 %add, 1
  %cmp = icmp ugt i8 %add, %N
  %base = select i1 %cmp, i16* %a, i16* %b
  %other = select i1 %cmp, i16* %b, i16* %b
  %cmp.1 = icmp eq i16* %base, %other
  br i1 %cmp.1, label %fail, label %loop

fail:
  br label %loop

loop:
  %ptr = phi i16* [ %base, %entry ], [ null, %fail ], [ %gep, %if.then ]
  %idx = phi i8 [ %and, %entry ], [ 0, %fail ], [ %inc, %if.then ]
  %undef = icmp eq i16* %ptr, undef
  br i1 %undef, label %exit, label %if.then

if.then:
  %load = load i16, i16* %ptr, align 2
  %inc = add nuw nsw i8 %idx, 1
  %gep = getelementptr inbounds i16, i16* %ptr, i8 %inc
  %cond = icmp eq i16* %gep, %other
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}

declare i8 @do_something_with_ptr(i8, i16*)

; CHECK-LABEL: call_pointer
; CHECK-NOT: uxt
define i8 @call_pointer(i8 zeroext %x, i8 zeroext %y, i16* %a, i16* %b) {
  %or = or i8 %x, %y
  %shr = lshr i8 %or, 1
  %add = add nuw i8 %shr, 2
  %cmp = icmp ne i8 %add, 0
  %ptr = select i1 %cmp, i16* %a, i16* %b
  %call = tail call zeroext i8 @do_something_with_ptr(i8 %shr, i16* %ptr)
  ret i8 %call
}

; CHECK-LABEL: pointer_to_pointer
; CHECK-NOT: uxt
define i16 @pointer_to_pointer(i16** %arg, i16 zeroext %limit) {
entry:
  %addr = load i16*, i16** %arg
  %val = load i16, i16* %addr
  %add = add nuw i16 %val, 7
  %cmp = icmp ult i16 %add, 256
  %res = select i1 %cmp, i16 128, i16 255
  ret i16 %res
}
