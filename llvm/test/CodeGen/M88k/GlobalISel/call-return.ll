; RUN: llc -mtriple=m88k -global-isel -stop-after=irtranslator -verify-machineinstrs -o - %s | FileCheck %s

; irtranslator, legalizer,regbankselect,instruction-select

; CHECK-LABLE: name: f1
; CHECK:       body:
; CHECK:         liveins: $r1
; CHECK:         RET implicit $r1
define void @f1() {
  ret void
}

define void @callf1() {
  call void() @f1()
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

define void @callf2() {
  call void(i32) @f2(i32 1)
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

define i32 @callf3() {
  %res = call i32(i32) @f3(i32 1)
  ret i32 %res
}

; CHECK-LABLE: name: f4
; CHECK:       body:
; CHECK:           liveins: $r1, $r2, $r3
; CHECK:           [[CP0:%[0-9]+]]:_(s32) = COPY $r2
; CHECK:           [[CP1:%[0-9]+]]:_(s32) = COPY $r3
; CHECK:           [[RES:%[0-9]+]]:_(s32) = G_AND [[CP0]], [[CP1]]
; CHECK:           $r2 = COPY [[RES]](s32)
; CHECK:           RET implicit $r1, implicit $r2
define i32 @f4(i32 %a, i32 %b) {
  %res = and i32 %a, %b
  ret i32 %res
}

define i32 @callf4() {
  %res = call i32(i32, i32) @f4(i32 1, i32 2)
  ret i32 %res
}

; TODO Is this IR correct?
define i16 @f5(i16 %a, i16 %b) {
  %res = and i16 %a, %b
  ret i16 %res
}


; CHECK-LABLE: name: f6
; CHECK:       body:
; CHECK:           liveins: $r1, $r2, $r3, $r4, $r5
; CHECK:           [[CP0:%[0-9]+]]:_(s32) = COPY $r2
; CHECK:           [[CP1:%[0-9]+]]:_(s32) = COPY $r3
; CHECK:           [[CP3:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[CP0]](s32), [[CP1]](s32)
; CHECK:           [[CP4:%[0-9]+]]:_(s32) = COPY $r4
; CHECK:           [[CP5:%[0-9]+]]:_(s32) = COPY $r5
; CHECK:           [[CP6:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[CP4]](s32), [[CP5]](s32)
; CHECK:           [[RES:%[0-9]+]]:_(s64) = G_AND [[CP3]], [[CP6]]
; CHECK:           [[UM1:%[0-9]+]]:_(s32), [[UM2:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[RES]](s64)
; CHECK:           $r2 = COPY [[UM1]](s32)
; CHECK:           $r3 = COPY [[UM2]](s32)
; CHECK:           RET implicit $r1, implicit $r2, implicit $r3
define i64 @f6(i64 %a, i64 %b) {
  %res = and i64 %a, %b
  ret i64 %res
}

define i64 @f7(i32 %a, i64 %b) {
  ret i64 %b
}

define float @f8(float %a, float %b) {
  %res = fadd float %a, %b
  ret float %res
}

; FIXME
;define double @f9(double %a, double %b) {
;  %res = fadd double %a, %b
;  ret double %res
;}


define i32 @f10(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h, i32 %i, i32 %j) {
  ret i32 %a
}

define i32 @callf10() {
  %res = call i32(i32,i32,i32,i32,i32,i32,i32,i32,i32,i32) @f10(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10)
  ret i32 %res
}
