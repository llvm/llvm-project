; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare void @somefunc(i32*)

; CHECK-LABEL: underalign:
; CHECK:      i32.load  $push[[L1:.+]]=, __stack_pointer{{.+}}
; CHECK-NEXT: i32.const $push[[L2:.+]]=, 16
; CHECK-NEXT: i32.sub   $push[[L10:.+]]=, $pop[[L1]], $pop[[L2]]
; CHECK-NEXT: tee_local $push{{.+}}=, $[[SP:.+]]=, $pop[[L10]]

; CHECK:      i32.add   $push[[underaligned:.+]]=, $[[SP]], $pop{{.+}}
; CHECK-NEXT: call      somefunc@FUNCTION, $pop[[underaligned]]

; CHECK:      i32.add   $push[[L5:.+]]=, $[[SP]], $pop{{.+}}
; CHECK-NEXT: i32.store __stack_pointer($pop{{.+}}), $pop[[L5]]
define void @underalign() {
entry:
  %underaligned = alloca i32, align 8
  call void @somefunc(i32* %underaligned)
  ret void
}

; CHECK-LABEL: overalign:
; CHECK:      i32.load   $push[[L10:.+]]=, __stack_pointer
; CHECK-NEXT: tee_local  $push[[L9:.+]]=, $[[BP:.+]]=, $pop[[L10]]
; CHECK-NEXT: i32.const  $push[[L2:.+]]=, 32
; CHECK-NEXT: i32.sub    $push[[L8:.+]]=, $pop[[L9]], $pop[[L2]]
; CHECK-NEXT: i32.const  $push[[L3:.+]]=, -32
; CHECK-NEXT: i32.and    $push[[L7:.+]]=, $pop[[L8]], $pop[[L3]]
; CHECK-NEXT: tee_local  $push{{.+}}=, $[[SP:.+]]=, $pop[[L7]]

; CHECK:      call       somefunc@FUNCTION, $[[SP]]

; CHECK:      copy_local $push[[L5:.+]]=, $[[BP]]
; CHECK-NEXT: i32.store  __stack_pointer($pop{{.+}}), $pop[[L5]]
define void @overalign() {
entry:
  %overaligned = alloca i32, align 32
  call void @somefunc(i32* %overaligned)
  ret void
}

; CHECK-LABEL: over_and_normal_align:
; CHECK:      i32.load   $push[[L14:.+]]=, __stack_pointer
; CHECK-NEXT: tee_local  $push[[L13:.+]]=, $[[BP:.+]]=, $pop[[L14]]
; CHECK:      i32.sub    $push[[L12:.+]]=, $pop[[L13]], $pop{{.+}}
; CHECK:      i32.and    $push[[L11:.+]]=, $pop[[L12]], $pop{{.+}}
; CHECK-NEXT: tee_local  $push{{.+}}=, $[[SP]]=, $pop[[L11]]

; CHECK:      i32.add    $push[[L6:.+]]=, $[[SP]], $pop{{.+}}
; CHECK-NEXT: call       somefunc@FUNCTION, $pop[[L6]]
; CHECK:      i32.add    $push[[L8:.+]]=, $[[SP]], $pop{{.+}}
; CHECK-NEXT: call       somefunc@FUNCTION, $pop[[L8]]

; CHECK:      copy_local $push[[L9:.+]]=, $[[BP]]
; CHECK-NEXT: i32.store  __stack_pointer({{.+}}), $pop[[L9]]
define void @over_and_normal_align() {
entry:
  %over = alloca i32, align 32
  %normal = alloca i32
  call void @somefunc(i32* %over)
  call void @somefunc(i32* %normal)
  ret void
}

; CHECK-LABEL: dynamic_overalign:
; CHECK:      i32.load   $push[[L18:.+]]=, __stack_pointer
; CHECK-NEXT: tee_local  $push[[L17:.+]]=, $[[SP:.+]]=, $pop[[L18]]
; CHECK-NEXT: copy_local $[[BP:.+]]=, $pop[[L17]]
; CHECK:      tee_local  $push{{.+}}=, $[[SP_2:.+]]=, $pop{{.+}}

; CHECK:      call       somefunc@FUNCTION, $[[SP_2]]

; CHECK: i32.store __stack_pointer($pop{{.+}}), $[[BP]]
define void @dynamic_overalign(i32 %num) {
entry:
  %dynamic = alloca i32, i32 %num, align 32
  call void @somefunc(i32* %dynamic)
  ret void
}

; CHECK-LABEL: overalign_and_dynamic:
; CHECK:      i32.load   $push[[L21:.+]]=, __stack_pointer
; CHECK-NEXT: tee_local  $push[[L20:.+]]=, $[[BP:.+]]=, $pop[[L21]]
; CHECK:      i32.sub    $push[[L19:.+]]=, $pop[[L20]], $pop{{.+}}
; CHECK:      i32.and    $push[[L18:.+]]=, $pop[[L19]], $pop{{.+}}
; CHECK:      tee_local  $push{{.+}}=, $[[FP:.+]]=, $pop[[L18]]
; CHECK:      i32.sub    $push[[L16:.+]]=, $[[FP]], $pop{{.+}}
; CHECK-NEXT: tee_local  $push{{.+}}=, $[[SP:.+]]=, $pop[[L16]]

; CHECK:      copy_local $push[[over:.+]]=, $[[FP]]
; CHECK-NEXT: call       somefunc@FUNCTION, $pop[[over]]
; CHECK-NEXT: call       somefunc@FUNCTION, $[[SP]]

; CHECK:      copy_local $push[[L12:.+]]=, $[[BP]]
; CHECK-NEXT: i32.store  __stack_pointer($pop{{.+}}), $pop[[L12]]
define void @overalign_and_dynamic(i32 %num) {
entry:
  %over = alloca i32, align 32
  %dynamic = alloca i32, i32 %num
  call void @somefunc(i32* %over)
  call void @somefunc(i32* %dynamic)
  ret void
}

; CHECK-LABEL: overalign_static_and_dynamic:
; CHECK:      i32.load   $push[[L26:.+]]=, __stack_pointer
; CHECK-NEXT: tee_local  $push[[L25:.+]]=, $[[BP:.+]]=, $pop[[L26]]
; CHECK:      i32.sub    $push[[L24:.+]]=, $pop[[L25]], $pop{{.+}}
; CHECK:      i32.and    $push[[L23:.+]]=, $pop[[L24]], $pop{{.+}}
; CHECK:      tee_local  $push{{.+}}=, $[[FP:.+]]=, $pop[[L23]]
; CHECK:      i32.sub    $push[[L21:.+]]=, $[[FP]], $pop{{.+}}
; CHECK-NEXT: tee_local  $push{{.+}}=, $[[SP:.+]]=, $pop[[L21]]

; CHECK:      copy_local $push[[L19:.+]]=, $[[FP]]
; CHECK:      tee_local  $push[[L18:.+]]=, $[[FP_2:.+]]=, $pop[[L19]]
; CHECK:      i32.add    $push[[over:.+]]=, $pop[[L18]], $pop{{.+}}
; CHECK-NEXT: call       somefunc@FUNCTION, $pop[[over]]
; CHECK:      call       somefunc@FUNCTION, $[[SP]]
; CHECK:      i32.add    $push[[static:.+]]=, $[[FP_2]], $pop{{.+}}
; CHECK-NEXT: call       somefunc@FUNCTION, $pop[[static]]

; CHECK:      copy_local $push[[L16:.+]]=, $[[BP]]
; CHECK-NEXT: i32.store  __stack_pointer({{.+}}), $pop[[L16]]
define void @overalign_static_and_dynamic(i32 %num) {
entry:
  %over = alloca i32, align 32
  %dynamic = alloca i32, i32 %num
  %static = alloca i32
  call void @somefunc(i32* %over)
  call void @somefunc(i32* %dynamic)
  call void @somefunc(i32* %static)
  ret void
}
