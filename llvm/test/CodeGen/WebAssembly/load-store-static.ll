; RUN: llc < %s -asm-verbose=false -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s -check-prefixes=NON-PIC,CHECK

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-emscripten"

@hidden_global         = external hidden global i32
@hidden_global_array   = external hidden global [10 x i32]
@external_global       = external dso_local global i32
@external_global_array = external dso_local global [10 x i32]

declare i32 @foo();

; For hidden symbols PIC code needs to offset all loads and stores
; by the value of the __memory_base global

define i32 @load_hidden_global() {
; CHECK-LABEL: load_hidden_global:
; NON-PIC:         i32.const $push0=, 0{{$}}
; NON-PIC-NEXT:    i32.load $push1=, hidden_global($pop0){{$}}
; CHECK-NEXT:    end_function

  %1 = load i32, i32* @hidden_global
  ret i32 %1
}

define i32 @load_hidden_global_offset() {
; CHECK-LABEL: load_hidden_global_offset:
; NON-PIC:     i32.const $push0=, 0{{$}}
; NON-PIC-NEXT:i32.load  $push1=, hidden_global_array+20($pop0){{$}}
; CHECK-NEXT:  end_function

  %1 = getelementptr [10 x i32], [10 x i32]* @hidden_global_array, i32 0, i32 5
  %2 = load i32, i32* %1
  ret i32 %2
}

; Store to a hidden global

define void @store_hidden_global(i32 %n) {
; CHECK-LABEL: store_hidden_global:
; NON-PIC:       i32.const $push0=, 0{{$}}
; NON-PIC-NEXT:  i32.store hidden_global($pop0), $0{{$}}
; CHECK-NEXT:    end_function

  store i32 %n, i32* @hidden_global
  ret void
}

define void @store_hidden_global_offset(i32 %n) {
; CHECK-LABEL: store_hidden_global_offset:
; NON-PIC:      i32.const $push0=, 0{{$}}
; NON-PIC-NEXT: i32.store hidden_global_array+20($pop0), $0{{$}}
; CHECK-NEXT:   end_function

  %1 = getelementptr [10 x i32], [10 x i32]* @hidden_global_array, i32 0, i32 5
  store i32 %n, i32* %1
  ret void
}

; For non-hidden globals PIC code has to load the address from a wasm global
; using the @GOT relocation type.


define i32 @load_external_global() {
; CHECK-LABEL:  load_external_global:
; NON-PIC:      i32.const $push0=, 0{{$}}
; NON-PIC-NEXT: i32.load $push1=, external_global($pop0){{$}}
; CHECK-NEXT:   end_function

  %1 = load i32, i32* @external_global
  ret i32 %1
}

define i32 @load_external_global_offset() {
; CHECK-LABEL:  load_external_global_offset:
; NON-PIC:      i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; NON-PIC-NEXT: i32.load $push{{[0-9]+}}=, external_global_array+20($pop[[L0]]){{$}}
; CHECK-NEXT:   end_function

  %1 = getelementptr [10 x i32], [10 x i32]* @external_global_array, i32 0, i32 5
  %2 = load i32, i32* %1
  ret i32 %2
}

; Store to a non-hidden global via the wasm global.

define void @store_external_global(i32 %n) {
; CHECK-LABEL:  store_external_global:
; NON-PIC:      i32.const $push0=, 0{{$}}
; NON-PIC-NEXT: i32.store external_global($pop0), $0{{$}}
; CHECK-NEXT:   end_function

  store i32 %n, i32* @external_global
  ret void
}

define void @store_external_global_offset(i32 %n) {
; CHECK-LABEL:  store_external_global_offset:
; NON-PIC:      i32.const $push0=, 0{{$}}
; NON-PIC-NEXT: i32.store external_global_array+20($pop0), $0{{$}}
; CHECK-NEXT:   end_function

  %1 = getelementptr [10 x i32], [10 x i32]* @external_global_array, i32 0, i32 5
  store i32 %n, i32* %1
  ret void
}
