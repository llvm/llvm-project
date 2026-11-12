; RUN: llc -mtriple=armv7-linux-gnueabihf %s -o - | FileCheck %s

; Check that we don't create an unpredictable STR instruction,
; e.g. str r0, [r0], #4

define ptr @earlyclobber-str-post(ptr %addr) nounwind {
; CHECK-LABEL: earlyclobber-str-post
; CHECK-NOT: str r[[REG:[0-9]+]], [r[[REG]]], #4
  %val = ptrtoint ptr %addr to i32
  store i32 %val, ptr %addr
  %new = getelementptr i32, ptr %addr, i32 1
  ret ptr %new
}

define ptr @earlyclobber-strh-post(ptr %addr) nounwind {
; CHECK-LABEL: earlyclobber-strh-post
; CHECK-NOT: strh r[[REG:[0-9]+]], [r[[REG]]], #2
  %val = ptrtoint ptr %addr to i32
  %tr = trunc i32 %val to i16
  store i16 %tr, ptr %addr
  %new = getelementptr i16, ptr %addr, i32 1
  ret ptr %new
}

define ptr @earlyclobber-strb-post(ptr %addr) nounwind {
; CHECK-LABEL: earlyclobber-strb-post
; CHECK-NOT: strb r[[REG:[0-9]+]], [r[[REG]]], #1
  %val = ptrtoint ptr %addr to i32
  %tr = trunc i32 %val to i8
  store i8 %tr, ptr %addr
  %new = getelementptr i8, ptr %addr, i32 1
  ret ptr %new
}
