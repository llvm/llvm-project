; RUN: llc < %s -asm-verbose=false -relocation-model=pic -fast-isel -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+atomics,+sign-ext | FileCheck %s
; RUN: llc < %s -asm-verbose=false -relocation-model=pic -fast-isel=false -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+atomics,+sign-ext | FileCheck %s

; Test that atomic operations in PIC mode.  Specifically we verify
; that atomic operations on global address load addres via @GOT or
; @MBREL relocations.

target triple = "wasm32-unknown-emscripten"

@external_global       = external        global i32
@hidden_global         = external hidden global i32

define i32 @rmw_add_external_global() {
; CHECK-LABEL: rmw_add_external_global:
; CHECK:         global.get $push[[L0:[0-9]+]]=, external_global@GOT{{$}}
; CHECK-NEXT:    i32.const $push[[L1:[0-9]+]]=, 42{{$}}
; CHECK-NEXT:    i32.atomic.rmw.add $push[[L2:[0-9]+]]=, 0($pop[[L0]]), $pop[[L1]]{{$}}
; CHECK-NEXT:    end_function
  %1 = atomicrmw add ptr @external_global, i32 42 seq_cst
  ret i32 %1
}

define i32 @rmw_add_hidden_global() {
; CHECK-LABEL: rmw_add_hidden_global:
; CHECK:         global.get $push[[L0:[0-9]+]]=, __memory_base{{$}}
; CHECK-NEXT:    i32.const $push[[L1:[0-9]+]]=, hidden_global@MBREL{{$}}
; CHECK-NEXT:    i32.add $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT:    i32.const $push[[L3:[0-9]+]]=, 42{{$}}
; CHECK-NEXT:    i32.atomic.rmw.add $push[[L4:[0-9]+]]=, 0($pop[[L2]]), $pop[[L3]]{{$}}
; CHECK-NEXT:    end_function
  %1 = atomicrmw add ptr @hidden_global, i32 42 seq_cst
  ret i32 %1
}
