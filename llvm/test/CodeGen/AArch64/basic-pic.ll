; RUN: llc -mtriple=aarch64-linux-gnu -verify-machineinstrs -relocation-model=pic %s -o - | FileCheck %s

@var = global i32 0

define i32 @get_globalvar() {
; CHECK-LABEL: get_globalvar:

  %val = load i32, i32* @var
; CHECK: adrp x[[GOTHI:[0-9]+]], :got:var
; CHECK: ldr x[[GOTLOC:[0-9]+]], [x[[GOTHI]], :got_lo12:var]
; CHECK: ldr w0, [x[[GOTLOC]]]

  ret i32 %val
}

define i32* @get_globalvaraddr() {
; CHECK-LABEL: get_globalvaraddr:

  %val = load i32, i32* @var
; CHECK: adrp x[[GOTHI:[0-9]+]], :got:var
; CHECK: ldr x0, [x[[GOTHI]], :got_lo12:var]

  ret i32* @var
}

@hiddenvar = hidden global i32 0

define i32 @get_hiddenvar() {
; CHECK-LABEL: get_hiddenvar:

  %val = load i32, i32* @hiddenvar
; CHECK: adrp x[[HI:[0-9]+]], hiddenvar
; CHECK: ldr w0, [x[[HI]], :lo12:hiddenvar]

  ret i32 %val
}

define i32* @get_hiddenvaraddr() {
; CHECK-LABEL: get_hiddenvaraddr:

  %val = load i32, i32* @hiddenvar
; CHECK: adrp [[HI:x[0-9]+]], hiddenvar
; CHECK: add x0, [[HI]], :lo12:hiddenvar

  ret i32* @hiddenvar
}

define void()* @get_func() {
; CHECK-LABEL: get_func:

  ret void()* bitcast(void()*()* @get_func to void()*)
; CHECK: adrp x[[GOTHI:[0-9]+]], :got:get_func
; CHECK: ldr x0, [x[[GOTHI]], :got_lo12:get_func]
}
