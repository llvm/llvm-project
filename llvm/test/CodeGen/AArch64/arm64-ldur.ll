; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s

define i64 @_f0(ptr %p) {
; CHECK: f0:
; CHECK: ldur x0, [x0, #-8]
; CHECK-NEXT: ret
  %tmp = getelementptr inbounds i64, ptr %p, i64 -1
  %ret = load i64, ptr %tmp, align 2
  ret i64 %ret
}
define i32 @_f1(ptr %p) {
; CHECK: f1:
; CHECK: ldur w0, [x0, #-4]
; CHECK-NEXT: ret
  %tmp = getelementptr inbounds i32, ptr %p, i64 -1
  %ret = load i32, ptr %tmp, align 2
  ret i32 %ret
}
define i16 @_f2(ptr %p) {
; CHECK: f2:
; CHECK: ldurh w0, [x0, #-2]
; CHECK-NEXT: ret
  %tmp = getelementptr inbounds i16, ptr %p, i64 -1
  %ret = load i16, ptr %tmp, align 2
  ret i16 %ret
}
define i8 @_f3(ptr %p) {
; CHECK: f3:
; CHECK: ldurb w0, [x0, #-1]
; CHECK-NEXT: ret
  %tmp = getelementptr inbounds i8, ptr %p, i64 -1
  %ret = load i8, ptr %tmp, align 2
  ret i8 %ret
}

define i64 @zext32(ptr %a) nounwind ssp {
; CHECK-LABEL: zext32:
; CHECK: ldur w0, [x0, #-12]
; CHECK-NEXT: ret
  %p = getelementptr inbounds i8, ptr %a, i64 -12
  %tmp2 = load i32, ptr %p, align 4
  %ret = zext i32 %tmp2 to i64

  ret i64 %ret
}
define i64 @zext16(ptr %a) nounwind ssp {
; CHECK-LABEL: zext16:
; CHECK: ldurh w0, [x0, #-12]
; CHECK-NEXT: ret
  %p = getelementptr inbounds i8, ptr %a, i64 -12
  %tmp2 = load i16, ptr %p, align 2
  %ret = zext i16 %tmp2 to i64

  ret i64 %ret
}
define i64 @zext8(ptr %a) nounwind ssp {
; CHECK-LABEL: zext8:
; CHECK: ldurb w0, [x0, #-12]
; CHECK-NEXT: ret
  %p = getelementptr inbounds i8, ptr %a, i64 -12
  %tmp2 = load i8, ptr %p, align 1
  %ret = zext i8 %tmp2 to i64

  ret i64 %ret
}
