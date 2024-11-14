; RUN: llc -mtriple=aarch64-linux-gnu -verify-machineinstrs -relocation-model=pic %s -o - | FileCheck %s

@var = global i32 0

define i32 @get_globalvar() {
; CHECK-LABEL: get_globalvar:
; CHECK:         adrp x[[GOTHI:[0-9]+]], :got:var
; CHECK-NEXT:    ldr x[[GOTLOC:[0-9]+]], [x[[GOTHI]], :got_lo12:var]
; CHECK-NEXT:    ldr w0, [x[[GOTLOC]]]

  %val = load i32, ptr @var
  ret i32 %val
}

define ptr @get_globalvaraddr() {
; CHECK-LABEL: get_globalvaraddr:
; CHECK:         adrp x[[GOTHI:[0-9]+]], :got:var
; CHECK-NEXT:    ldr x0, [x[[GOTHI]], :got_lo12:var]

  %val = load i32, ptr @var
  ret ptr @var
}
