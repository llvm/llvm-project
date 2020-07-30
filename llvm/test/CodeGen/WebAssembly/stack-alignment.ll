; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers | FileCheck -DPTR=32 %s
; RUN: llc < %s --mtriple=wasm64-unknown-unknown -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers | FileCheck -DPTR=64 %s

declare void @somefunc(i32*)

; CHECK-LABEL: underalign:
; CHECK:      global.get $push[[L1:.+]]=, __stack_pointer{{$}}
; CHECK-NEXT: i[[PTR]].const $push[[L2:.+]]=, 16
; CHECK-NEXT: i[[PTR]].sub $push[[L10:.+]]=, $pop[[L1]], $pop[[L2]]
; CHECK-NEXT: local.tee $push{{.+}}=, [[SP:.+]], $pop[[L10]]

; CHECK:      local.get $push[[L3:.+]]=, [[SP]]{{$}}
; CHECK:      i[[PTR]].add $push[[underaligned:.+]]=, $pop[[L3]], $pop{{.+}}
; CHECK-NEXT: call somefunc, $pop[[underaligned]]

; CHECK:      local.get $push[[M4:.+]]=, [[SP]]{{$}}
; CHECK:      i[[PTR]].add $push[[L5:.+]]=, $pop[[M4]], $pop{{.+}}
; CHECK-NEXT: global.set __stack_pointer, $pop[[L5]]
define void @underalign() {
entry:
  %underaligned = alloca i32, align 8
  call void @somefunc(i32* %underaligned)
  ret void
}

; CHECK-LABEL: overalign:
; CHECK:      global.get $push[[L10:.+]]=, __stack_pointer{{$}}
; CHECK-NEXT: local.tee $push[[L9:.+]]=, [[BP:.+]], $pop[[L10]]
; CHECK-NEXT: i[[PTR]].const $push[[L2:.+]]=, 32
; CHECK-NEXT: i[[PTR]].sub $push[[L8:.+]]=, $pop[[L9]], $pop[[L2]]
; CHECK-NEXT: i[[PTR]].const $push[[L3:.+]]=, -32
; CHECK-NEXT: i[[PTR]].and $push[[L7:.+]]=, $pop[[L8]], $pop[[L3]]
; CHECK-NEXT: local.tee $push{{.+}}=, [[SP:.+]], $pop[[L7]]

; CHECK:      local.get $push[[M5:.+]]=, [[SP]]{{$}}
; CHECK:      call somefunc, $pop[[M5]]{{$}}

; CHECK:      local.get $push[[M6:.+]]=, [[BP]]{{$}}
; CHECK-NEXT: global.set __stack_pointer, $pop[[M6]]
define void @overalign() {
entry:
  %overaligned = alloca i32, align 32
  call void @somefunc(i32* %overaligned)
  ret void
}

; CHECK-LABEL: over_and_normal_align:
; CHECK:      global.get $push[[L14:.+]]=, __stack_pointer{{$}}
; CHECK-NEXT: local.tee $push[[L13:.+]]=, [[BP:.+]], $pop[[L14]]
; CHECK:      i[[PTR]].sub $push[[L12:.+]]=, $pop[[L13]], $pop{{.+}}
; CHECK:      i[[PTR]].and $push[[L11:.+]]=, $pop[[L12]], $pop{{.+}}
; CHECK-NEXT: local.tee $push{{.+}}=, [[SP:.+]], $pop[[L11]]

; CHECK:      local.get  $push[[M6:.+]]=, [[SP]]{{$}}
; CHECK:      i[[PTR]].add $push[[L6:.+]]=, $pop[[M6]], $pop{{.+}}
; CHECK-NEXT: call somefunc, $pop[[L6]]
; CHECK:      local.get $push[[M7:.+]]=, [[SP]]{{$}}
; CHECK:      i[[PTR]].add $push[[L8:.+]]=, $pop[[M7]], $pop{{.+}}
; CHECK-NEXT: call somefunc, $pop[[L8]]

; CHECK:      local.get $push[[L6:.+]]=, [[BP]]{{$}}
; CHECK-NEXT: global.set __stack_pointer, $pop[[L6]]
define void @over_and_normal_align() {
entry:
  %over = alloca i32, align 32
  %normal = alloca i32
  call void @somefunc(i32* %over)
  call void @somefunc(i32* %normal)
  ret void
}

; CHECK-LABEL: dynamic_overalign:
; CHECK:      global.get $push[[L18:.+]]=, __stack_pointer{{$}}
; CHECK-NEXT: local.tee $push[[L17:.+]]=, [[SP:.+]], $pop[[L18]]
; CHECK-NEXT: local.set [[BP:.+]], $pop[[L17]]
; CHECK:      local.tee $push{{.+}}=, [[SP_2:.+]], $pop{{.+}}

; CHECK:      local.get $push[[M8:.+]]=, [[SP_2]]{{$}}
; CHECK:      call somefunc, $pop[[M8]]

; CHECK:      local.get $push[[M9:.+]]=, [[BP]]{{$}}
; CHECK-NEXT: global.set __stack_pointer, $pop[[M9]]
define void @dynamic_overalign(i32 %num) {
entry:
  %dynamic = alloca i32, i32 %num, align 32
  call void @somefunc(i32* %dynamic)
  ret void
}

; CHECK-LABEL: overalign_and_dynamic:
; CHECK:      global.get $push[[L21:.+]]=, __stack_pointer{{$}}
; CHECK-NEXT: local.tee $push[[L20:.+]]=, [[BP:.+]], $pop[[L21]]
; CHECK:      i[[PTR]].sub $push[[L19:.+]]=, $pop[[L20]], $pop{{.+}}
; CHECK:      i[[PTR]].and $push[[L18:.+]]=, $pop[[L19]], $pop{{.+}}
; CHECK:      local.tee $push{{.+}}=, [[FP:.+]], $pop[[L18]]
; CHECK:      local.get $push[[M10:.+]]=, [[FP]]{{$}}
; CHECK:      i[[PTR]].sub $push[[L16:.+]]=, $pop[[M10]], $pop{{.+}}
; CHECK-NEXT: local.tee $push{{.+}}=, [[SP:.+]], $pop[[L16]]

; CHECK:      local.get $push[[over:.+]]=, [[FP]]
; CHECK-NEXT: call somefunc, $pop[[over]]
; CHECK:      local.get $push[[another:.+]]=, [[SP]]
; CHECK-NEXT: call somefunc, $pop[[another]]

; CHECK:      local.get  $push[[M11:.+]]=, [[BP]]{{$}}
; CHECK-NEXT: global.set __stack_pointer, $pop[[M11]]
define void @overalign_and_dynamic(i32 %num) {
entry:
  %over = alloca i32, align 32
  %dynamic = alloca i32, i32 %num
  call void @somefunc(i32* %over)
  call void @somefunc(i32* %dynamic)
  ret void
}

; CHECK-LABEL: overalign_static_and_dynamic:
; CHECK:      global.get $push[[L26:.+]]=, __stack_pointer{{$}}
; CHECK-NEXT: local.tee $push[[L25:.+]]=, [[BP:.+]], $pop[[L26]]
; CHECK:      i[[PTR]].sub $push[[L24:.+]]=, $pop[[L25]], $pop{{.+}}
; CHECK:      i[[PTR]].and $push[[L23:.+]]=, $pop[[L24]], $pop{{.+}}
; CHECK:      local.tee $push{{.+}}=, [[FP:.+]], $pop[[L23]]
; CHECK:      local.get $push[[M12:.+]]=, [[FP]]{{$}}
; CHECK:      i[[PTR]].sub $push[[L21:.+]]=, $pop[[M12]], $pop{{.+}}
; CHECK-NEXT: local.tee $push{{.+}}=, [[SP:.+]], $pop[[L21]]

; CHECK:      local.get $push[[L19:.+]]=, [[FP]]
; CHECK:      local.tee $push[[L18:.+]]=, [[FP_2:.+]], $pop[[L19]]
; CHECK:      i[[PTR]].add $push[[over:.+]]=, $pop[[L18]], $pop{{.+}}
; CHECK-NEXT: call somefunc, $pop[[over]]
; CHECK:      local.get $push[[M12:.+]]=, [[SP]]
; CHECK:      call somefunc, $pop[[M12]]
; CHECK:      local.get $push[[M13:.+]]=, [[FP_2]]
; CHECK:      i[[PTR]].add $push[[static:.+]]=, $pop[[M13]], $pop{{.+}}
; CHECK-NEXT: call somefunc, $pop[[static]]

; CHECK:      local.get  $push[[M14:.+]]=, [[BP]]{{$}}
; CHECK-NEXT: global.set __stack_pointer, $pop[[M14]]
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
