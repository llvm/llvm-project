; RUN: llc -mattr=avr6,sram < %s -march=avr | FileCheck %s

define void @store8(ptr %x, i8 %y) {
; CHECK-LABEL: store8:
; CHECK: st {{[XYZ]}}, r22
  store i8 %y, ptr %x
  ret void
}

define void @store16(ptr %x, i16 %y) {
; CHECK-LABEL: store16:
; CHECK: std {{[YZ]}}+1, r23
; CHECK: st {{[YZ]}}, r22
  store i16 %y, ptr %x
  ret void
}

define void @store8disp(ptr %x, i8 %y) {
; CHECK-LABEL: store8disp:
; CHECK: std {{[YZ]}}+63, r22
  %arrayidx = getelementptr inbounds i8, ptr %x, i16 63
  store i8 %y, ptr %arrayidx
  ret void
}

define void @store8nodisp(ptr %x, i8 %y) {
; CHECK-LABEL: store8nodisp:
; CHECK: movw r26, r24
; CHECK: subi r26, 192
; CHECK: sbci r27, 255
; CHECK: st {{[XYZ]}}, r22
  %arrayidx = getelementptr inbounds i8, ptr %x, i16 64
  store i8 %y, ptr %arrayidx
  ret void
}

define void @store16disp(ptr %x, i16 %y) {
; CHECK-LABEL: store16disp:
; CHECK: std {{[YZ]}}+63, r23
; CHECK: std {{[YZ]}}+62, r22
  %arrayidx = getelementptr inbounds i16, ptr %x, i16 31
  store i16 %y, ptr %arrayidx
  ret void
}

define void @store16nodisp(ptr %x, i16 %y) {
; CHECK-LABEL: store16nodisp:
; CHECK: subi r24, 192
; CHECK: sbci r25, 255
; CHECK: movw r30, r24
; CHECK: std {{[YZ]}}+1, r23
; CHECK: st {{[YZ]}}, r22
  %arrayidx = getelementptr inbounds i16, ptr %x, i16 32
  store i16 %y, ptr %arrayidx
  ret void
}

define void @store8postinc(ptr %x, i8 %y) {
; CHECK-LABEL: store8postinc:
; CHECK: st {{[XYZ]}}+, {{.*}}
entry:
  %tobool3 = icmp eq i8 %y, 0
  br i1 %tobool3, label %while.end, label %while.body
while.body:                                       ; preds = %entry, %while.body
  %dec5.in = phi i8 [ %dec5, %while.body ], [ %y, %entry ]
  %x.addr.04 = phi ptr [ %incdec.ptr, %while.body ], [ %x, %entry ]
  %dec5 = add i8 %dec5.in, -1
  %incdec.ptr = getelementptr inbounds i8, ptr %x.addr.04, i16 1
  store i8 %dec5, ptr %x.addr.04
  %tobool = icmp eq i8 %dec5, 0
  br i1 %tobool, label %while.end, label %while.body
while.end:                                        ; preds = %while.body, %entry
  ret void
}

define void @store16postinc(ptr %x, i16 %y) {
; CHECK-LABEL: store16postinc:
; CHECK: std {{[XYZ]}}+1, {{.*}}
; CHECK: st  {{[XYZ]}}, {{.*}}
entry:
  %tobool3 = icmp eq i16 %y, 0
  br i1 %tobool3, label %while.end, label %while.body
while.body:                                       ; preds = %entry, %while.body
  %dec5.in = phi i16 [ %dec5, %while.body ], [ %y, %entry ]
  %x.addr.04 = phi ptr [ %incdec.ptr, %while.body ], [ %x, %entry ]
  %dec5 = add nsw i16 %dec5.in, -1
  %incdec.ptr = getelementptr inbounds i16, ptr %x.addr.04, i16 1
  store i16 %dec5, ptr %x.addr.04
  %tobool = icmp eq i16 %dec5, 0
  br i1 %tobool, label %while.end, label %while.body
while.end:                                        ; preds = %while.body, %entry
  ret void
}

define void @store8predec(ptr %x, i8 %y) {
; CHECK-LABEL: store8predec:
; CHECK: st -{{[XYZ]}}, {{.*}}
entry:
  %tobool3 = icmp eq i8 %y, 0
  br i1 %tobool3, label %while.end, label %while.body
while.body:                                       ; preds = %entry, %while.body
  %dec5.in = phi i8 [ %dec5, %while.body ], [ %y, %entry ]
  %x.addr.04 = phi ptr [ %incdec.ptr, %while.body ], [ %x, %entry ]
  %dec5 = add i8 %dec5.in, -1
  %incdec.ptr = getelementptr inbounds i8, ptr %x.addr.04, i16 -1
  store i8 %dec5, ptr %incdec.ptr
  %tobool = icmp eq i8 %dec5, 0
  br i1 %tobool, label %while.end, label %while.body
while.end:                                        ; preds = %while.body, %entry
  ret void
}

define void @store16predec(ptr %x, i16 %y) {
; CHECK-LABEL: store16predec:
; CHECK: st -{{[XYZ]}}, {{.*}}
; CHECK: st -{{[XYZ]}}, {{.*}}
entry:
  %tobool3 = icmp eq i16 %y, 0
  br i1 %tobool3, label %while.end, label %while.body
while.body:                                       ; preds = %entry, %while.body
  %dec5.in = phi i16 [ %dec5, %while.body ], [ %y, %entry ]
  %x.addr.04 = phi ptr [ %incdec.ptr, %while.body ], [ %x, %entry ]
  %dec5 = add nsw i16 %dec5.in, -1
  %incdec.ptr = getelementptr inbounds i16, ptr %x.addr.04, i16 -1
  store i16 %dec5, ptr %incdec.ptr
  %tobool = icmp eq i16 %dec5, 0
  br i1 %tobool, label %while.end, label %while.body
while.end:                                        ; preds = %while.body, %entry
  ret void
}
