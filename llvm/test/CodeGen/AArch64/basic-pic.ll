; RUN: llc -mtriple=aarch64-linux-gnu -verify-machineinstrs -relocation-model=pic %s -o - | FileCheck %s

@var = global i32 0

define i32 @get_globalvar() {
; CHECK-LABEL: get_globalvar:

  %val = load i32, ptr @var
; CHECK: adrp x[[GOTHI:[0-9]+]], :got:var
; CHECK: ldr x[[GOTLOC:[0-9]+]], [x[[GOTHI]], :got_lo12:var]
; CHECK: ldr w0, [x[[GOTLOC]]]

  ret i32 %val
}

define ptr @get_globalvaraddr() {
; CHECK-LABEL: get_globalvaraddr:

  %val = load i32, ptr @var
; CHECK: adrp x[[GOTHI:[0-9]+]], :got:var
; CHECK: ldr x0, [x[[GOTHI]], :got_lo12:var]

  ret ptr @var
}
