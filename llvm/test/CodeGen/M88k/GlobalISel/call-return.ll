; RUN: llc -mtriple=m88k -global-isel -stop-after=irtranslator -verify-machineinstrs -o - %s | FileCheck %s

; CHECK-LABLE: name: f1
; CHECK:       body:
; CHECK:         liveins: $r1
; CHECK:         RET implicit $r1
define void @f1() {
  ret void
}

; CHECK-LABLE: name: f2
; CHECK:       body:
; CHECK:         liveins: $r1, $r2
; CHECK:         [[CP:%[0-9]+]]:_(s32) = COPY $r2
; CHECK:         RET implicit $r1
define void @f2(i32 %a) {
  ret void
}

; CHECK-LABLE: name: f3
; CHECK:       body:
; CHECK:         liveins: $r1, $r2
; CHECK:         [[CP:%[0-9]+]]:_(s32) = COPY $r2
; CHECK:         $r2 = COPY [[CP]](s32)
; CHECK:         RET implicit $r1, implicit $r2
define i32 @f3(i32 %a) {
  ret i32 %a
}

define i32 @f4(i32 %a, i32 %b) {
  %sum = add i32 %a, %b
  ret i32 %sum
}
