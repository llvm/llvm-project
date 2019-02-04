; RUN: llc -mattr=avr6,sram < %s -march=avr | FileCheck %s

define i8 @load8(i8* %x) {
; CHECK-LABEL: load8:
; CHECK: ld r24, {{[XYZ]}}
  %1 = load i8, i8* %x
  ret i8 %1
}

define i16 @load16(i16* %x) {
; CHECK-LABEL: load16:
; CHECK: ld r24,  [[PTR:[XYZ]]]
; CHECK: ldd r25, [[PTR]]+1
  %1 = load i16, i16* %x
  ret i16 %1
}

define i8 @load8disp(i8* %x) {
; CHECK-LABEL: load8disp:
; CHECK: ldd r24, {{[YZ]}}+63
  %1 = getelementptr inbounds i8, i8* %x, i64 63
  %2 = load i8, i8* %1
  ret i8 %2
}

define i8 @load8nodisp(i8* %x) {
; CHECK-LABEL: load8nodisp:
; CHECK: movw r26, r24
; CHECK: subi r26, 192
; CHECK: sbci r27, 255
; CHECK: ld r24, {{[XYZ]}}
  %1 = getelementptr inbounds i8, i8* %x, i64 64
  %2 = load i8, i8* %1
  ret i8 %2
}

define i16 @load16disp(i16* %x) {
; CHECK-LABEL: load16disp:
; CHECK: ldd r24, [[PTR:[YZ]]]+62
; CHECK: ldd r25, [[PTR]]+63
  %1 = getelementptr inbounds i16, i16* %x, i64 31
  %2 = load i16, i16* %1
  ret i16 %2
}

define i16 @load16nodisp(i16* %x) {
; CHECK-LABEL: load16nodisp:
; CHECK: movw r26, r24
; CHECK: subi r26, 192
; CHECK: sbci r27, 255
; CHECK: ld r24,  [[PTR:[XYZ]]]
; CHECK: ldd r25, [[PTR]]+1
  %1 = getelementptr inbounds i16, i16* %x, i64 32
  %2 = load i16, i16* %1
  ret i16 %2
}

define i8 @load8postinc(i8* %x, i8 %y) {
; CHECK-LABEL: load8postinc:
; CHECK: ld {{.*}}, {{[XYZ]}}+
entry:
  %tobool6 = icmp eq i8 %y, 0
  br i1 %tobool6, label %while.end, label %while.body
while.body:                                       ; preds = %entry, %while.body
  %r.09 = phi i8 [ %add, %while.body ], [ 0, %entry ]
  %y.addr.08 = phi i8 [ %dec, %while.body ], [ %y, %entry ]
  %x.addr.07 = phi i8* [ %incdec.ptr, %while.body ], [ %x, %entry ]
  %dec = add i8 %y.addr.08, -1
  %incdec.ptr = getelementptr inbounds i8, i8* %x.addr.07, i16 1
  %0 = load i8, i8* %x.addr.07
  %add = add i8 %0, %r.09
  %tobool = icmp eq i8 %dec, 0
  br i1 %tobool, label %while.end, label %while.body
while.end:                                        ; preds = %while.body, %entry
  %r.0.lcssa = phi i8 [ 0, %entry ], [ %add, %while.body ]
  ret i8 %r.0.lcssa
}

define i16 @load16postinc(i16* %x, i16 %y) {
; CHECK-LABEL: load16postinc:
; CHECK: ld {{.*}}, {{[XYZ]}}+
; CHECK: ld {{.*}}, {{[XYZ]}}+
entry:
  %tobool2 = icmp eq i16 %y, 0
  br i1 %tobool2, label %while.end, label %while.body
while.body:                                       ; preds = %entry, %while.body
  %r.05 = phi i16 [ %add, %while.body ], [ 0, %entry ]
  %y.addr.04 = phi i16 [ %dec, %while.body ], [ %y, %entry ]
  %x.addr.03 = phi i16* [ %incdec.ptr, %while.body ], [ %x, %entry ]
  %dec = add nsw i16 %y.addr.04, -1
  %incdec.ptr = getelementptr inbounds i16, i16* %x.addr.03, i16 1
  %0 = load i16, i16* %x.addr.03
  %add = add nsw i16 %0, %r.05
  %tobool = icmp eq i16 %dec, 0
  br i1 %tobool, label %while.end, label %while.body
while.end:                                        ; preds = %while.body, %entry
  %r.0.lcssa = phi i16 [ 0, %entry ], [ %add, %while.body ]
  ret i16 %r.0.lcssa
}

define i8 @load8predec(i8* %x, i8 %y) {
; CHECK-LABEL: load8predec:
; CHECK: ld {{.*}}, -{{[XYZ]}}
entry:
  %tobool6 = icmp eq i8 %y, 0
  br i1 %tobool6, label %while.end, label %while.body
while.body:                                       ; preds = %entry, %while.body
  %r.09 = phi i8 [ %add, %while.body ], [ 0, %entry ]
  %y.addr.08 = phi i8 [ %dec, %while.body ], [ %y, %entry ]
  %x.addr.07 = phi i8* [ %incdec.ptr, %while.body ], [ %x, %entry ]
  %dec = add i8 %y.addr.08, -1
  %incdec.ptr = getelementptr inbounds i8, i8* %x.addr.07, i16 -1
  %0 = load i8, i8* %incdec.ptr
  %add = add i8 %0, %r.09
  %tobool = icmp eq i8 %dec, 0
  br i1 %tobool, label %while.end, label %while.body
while.end:                                        ; preds = %while.body, %entry
  %r.0.lcssa = phi i8 [ 0, %entry ], [ %add, %while.body ]
  ret i8 %r.0.lcssa
}

define i16 @load16predec(i16* %x, i16 %y) {
; CHECK-LABEL: load16predec:
; CHECK: ld {{.*}}, -{{[XYZ]}}
; CHECK: ld {{.*}}, -{{[XYZ]}}
entry:
  %tobool2 = icmp eq i16 %y, 0
  br i1 %tobool2, label %while.end, label %while.body
while.body:                                       ; preds = %entry, %while.body
  %r.05 = phi i16 [ %add, %while.body ], [ 0, %entry ]
  %y.addr.04 = phi i16 [ %dec, %while.body ], [ %y, %entry ]
  %x.addr.03 = phi i16* [ %incdec.ptr, %while.body ], [ %x, %entry ]
  %dec = add nsw i16 %y.addr.04, -1
  %incdec.ptr = getelementptr inbounds i16, i16* %x.addr.03, i16 -1
  %0 = load i16, i16* %incdec.ptr
  %add = add nsw i16 %0, %r.05
  %tobool = icmp eq i16 %dec, 0
  br i1 %tobool, label %while.end, label %while.body
while.end:                                        ; preds = %while.body, %entry
  %r.0.lcssa = phi i16 [ 0, %entry ], [ %add, %while.body ]
  ret i16 %r.0.lcssa
}
