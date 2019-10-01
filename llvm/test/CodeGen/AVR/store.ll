; RUN: llc -mattr=avr6,sram < %s -march=avr | FileCheck %s

define void @store8(i8* %x, i8 %y) {
; CHECK-LABEL: store8:
; CHECK: st {{[XYZ]}}, r22
  store i8 %y, i8* %x
  ret void
}

define void @store16(i16* %x, i16 %y) {
; CHECK-LABEL: store16:
; CHECK: st {{[YZ]}}, r22
; CHECK: std {{[YZ]}}+1, r23
  store i16 %y, i16* %x
  ret void
}

define void @store8disp(i8* %x, i8 %y) {
; CHECK-LABEL: store8disp:
; CHECK: std {{[YZ]}}+63, r22
  %arrayidx = getelementptr inbounds i8, i8* %x, i16 63
  store i8 %y, i8* %arrayidx
  ret void
}

define void @store8nodisp(i8* %x, i8 %y) {
; CHECK-LABEL: store8nodisp:
; CHECK: movw r26, r24
; CHECK: subi r26, 192
; CHECK: sbci r27, 255
; CHECK: st {{[XYZ]}}, r22
  %arrayidx = getelementptr inbounds i8, i8* %x, i16 64
  store i8 %y, i8* %arrayidx
  ret void
}

define void @store16disp(i16* %x, i16 %y) {
; CHECK-LABEL: store16disp:
; CHECK: std {{[YZ]}}+62, r22
; CHECK: std {{[YZ]}}+63, r23
  %arrayidx = getelementptr inbounds i16, i16* %x, i16 31
  store i16 %y, i16* %arrayidx
  ret void
}

define void @store16nodisp(i16* %x, i16 %y) {
; CHECK-LABEL: store16nodisp:
; CHECK: subi r24, 192
; CHECK: sbci r25, 255
; CHECK: movw r30, r24
; CHECK: st {{[YZ]}}, r22
; CHECK: std {{[YZ]}}+1, r23
  %arrayidx = getelementptr inbounds i16, i16* %x, i16 32
  store i16 %y, i16* %arrayidx
  ret void
}

define void @store8postinc(i8* %x, i8 %y) {
; CHECK-LABEL: store8postinc:
; CHECK: st {{[XYZ]}}+, {{.*}}
entry:
  %tobool3 = icmp eq i8 %y, 0
  br i1 %tobool3, label %while.end, label %while.body
while.body:                                       ; preds = %entry, %while.body
  %dec5.in = phi i8 [ %dec5, %while.body ], [ %y, %entry ]
  %x.addr.04 = phi i8* [ %incdec.ptr, %while.body ], [ %x, %entry ]
  %dec5 = add i8 %dec5.in, -1
  %incdec.ptr = getelementptr inbounds i8, i8* %x.addr.04, i16 1
  store i8 %dec5, i8* %x.addr.04
  %tobool = icmp eq i8 %dec5, 0
  br i1 %tobool, label %while.end, label %while.body
while.end:                                        ; preds = %while.body, %entry
  ret void
}

define void @store16postinc(i16* %x, i16 %y) {
; CHECK-LABEL: store16postinc:
; CHECK: st {{[XYZ]}}+, {{.*}}
; CHECK: st {{[XYZ]}}+, {{.*}}
entry:
  %tobool3 = icmp eq i16 %y, 0
  br i1 %tobool3, label %while.end, label %while.body
while.body:                                       ; preds = %entry, %while.body
  %dec5.in = phi i16 [ %dec5, %while.body ], [ %y, %entry ]
  %x.addr.04 = phi i16* [ %incdec.ptr, %while.body ], [ %x, %entry ]
  %dec5 = add nsw i16 %dec5.in, -1
  %incdec.ptr = getelementptr inbounds i16, i16* %x.addr.04, i16 1
  store i16 %dec5, i16* %x.addr.04
  %tobool = icmp eq i16 %dec5, 0
  br i1 %tobool, label %while.end, label %while.body
while.end:                                        ; preds = %while.body, %entry
  ret void
}

define void @store8predec(i8* %x, i8 %y) {
; CHECK-LABEL: store8predec:
; CHECK: st -{{[XYZ]}}, {{.*}}
entry:
  %tobool3 = icmp eq i8 %y, 0
  br i1 %tobool3, label %while.end, label %while.body
while.body:                                       ; preds = %entry, %while.body
  %dec5.in = phi i8 [ %dec5, %while.body ], [ %y, %entry ]
  %x.addr.04 = phi i8* [ %incdec.ptr, %while.body ], [ %x, %entry ]
  %dec5 = add i8 %dec5.in, -1
  %incdec.ptr = getelementptr inbounds i8, i8* %x.addr.04, i16 -1
  store i8 %dec5, i8* %incdec.ptr
  %tobool = icmp eq i8 %dec5, 0
  br i1 %tobool, label %while.end, label %while.body
while.end:                                        ; preds = %while.body, %entry
  ret void
}

define void @store16predec(i16* %x, i16 %y) {
; CHECK-LABEL: store16predec:
; CHECK: st -{{[XYZ]}}, {{.*}}
; CHECK: st -{{[XYZ]}}, {{.*}}
entry:
  %tobool3 = icmp eq i16 %y, 0
  br i1 %tobool3, label %while.end, label %while.body
while.body:                                       ; preds = %entry, %while.body
  %dec5.in = phi i16 [ %dec5, %while.body ], [ %y, %entry ]
  %x.addr.04 = phi i16* [ %incdec.ptr, %while.body ], [ %x, %entry ]
  %dec5 = add nsw i16 %dec5.in, -1
  %incdec.ptr = getelementptr inbounds i16, i16* %x.addr.04, i16 -1
  store i16 %dec5, i16* %incdec.ptr
  %tobool = icmp eq i16 %dec5, 0
  br i1 %tobool, label %while.end, label %while.body
while.end:                                        ; preds = %while.body, %entry
  ret void
}
