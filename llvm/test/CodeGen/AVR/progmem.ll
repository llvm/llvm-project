; RUN: llc < %s -march=avr -mattr=movw,lpm | FileCheck %s
; XFAIL: *

; Tests the standard LPM instruction

define i8 @test8(i8 addrspace(1)* %p) {
; CHECK-LABEL: test8:
; CHECK: movw r30, r24
; CHECK: lpm r24, Z
  %1 = load i8, i8 addrspace(1)* %p
  ret i8 %1
}

define i16 @test16(i16 addrspace(1)* %p) {
; CHECK-LABEL: test16:
; CHECK: movw r30, r24
; CHECK: lpm r24, Z+
; CHECK: lpm r25, Z+
  %1 = load i16, i16 addrspace(1)* %p
  ret i16 %1
}

define i8 @test8postinc(i8 addrspace(1)* %x, i8 %y) {
; CHECK-LABEL: test8postinc:
; CHECK: movw r30, r24
; CHECK: lpm {{.*}}, Z+
entry:
  %cmp10 = icmp sgt i8 %y, 0
  br i1 %cmp10, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %ret.013 = phi i8 [ %add, %for.body ], [ 0, %entry ]
  %i.012 = phi i8 [ %inc, %for.body ], [ 0, %entry ]
  %x.addr.011 = phi i8 addrspace(1)* [ %incdec.ptr, %for.body ], [ %x, %entry ]
  %incdec.ptr = getelementptr inbounds i8, i8 addrspace(1)* %x.addr.011, i16 1
  %0 = load i8, i8 addrspace(1)* %x.addr.011
  %add = add i8 %0, %ret.013
  %inc = add i8 %i.012, 1
  %exitcond = icmp eq i8 %inc, %y
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %ret.0.lcssa = phi i8 [ 0, %entry ], [ %add, %for.body ]
  ret i8 %ret.0.lcssa
}

define i16 @test16postinc(i16 addrspace(1)* %x, i8 %y) {
; CHECK-LABEL: test16postinc:
; CHECK: movw r30, r24
; CHECK: lpm {{.*}}, Z+
; CHECK: lpm {{.*}}, Z+
entry:
  %cmp5 = icmp sgt i8 %y, 0
  br i1 %cmp5, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %ret.08 = phi i16 [ %add, %for.body ], [ 0, %entry ]
  %i.07 = phi i8 [ %inc, %for.body ], [ 0, %entry ]
  %x.addr.06 = phi i16 addrspace(1)* [ %incdec.ptr, %for.body ], [ %x, %entry ]
  %incdec.ptr = getelementptr inbounds i16, i16 addrspace(1)* %x.addr.06, i16 1
  %0 = load i16, i16 addrspace(1)* %x.addr.06
  %add = add nsw i16 %0, %ret.08
  %inc = add i8 %i.07, 1
  %exitcond = icmp eq i8 %inc, %y
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %ret.0.lcssa = phi i16 [ 0, %entry ], [ %add, %for.body ]
  ret i16 %ret.0.lcssa
}
