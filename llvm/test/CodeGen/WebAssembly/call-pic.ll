; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers -relocation-model=pic -fast-isel=1 | FileCheck %s
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers -relocation-model=pic -fast-isel=0 | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-emscripten"

declare i32 @foo()
declare i32 @bar()
declare hidden i32 @hidden_function()

@indirect_func = hidden global i32 ()* @foo
@alias_func = hidden alias i32 (), i32 ()* @local_function

define i32 @local_function() {
  ret i32 1
}

define void @call_indirect_func() {
; CHECK-LABEL: call_indirect_func:
; CHECK:      global.get $push[[L0:[0-9]+]]=, __memory_base{{$}}
; CHECK-NEXT: i32.const $push[[L1:[0-9]+]]=, indirect_func@MBREL{{$}}
; CHECK-NEXT: i32.add $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: i32.load $push[[L3:[0-9]+]]=, 0($pop[[L2]]){{$}}
; CHECK-NEXT: call_indirect $push[[L4:[0-9]+]]=, $pop[[L3]]{{$}}
  %1 = load i32 ()*, i32 ()** @indirect_func, align 4
  %call = call i32 %1()
  ret void
}

define void @call_direct() {
; CHECK-LABEL: call_direct:
; CHECK: .functype call_direct () -> ()
; CHECK-NEXT: call $push0=, foo{{$}}
; CHECK-NEXT: drop $pop0{{$}}
; CHECK-NEXT: return{{$}}
  %call = call i32 @foo()
  ret void
}

define void @call_alias_func() {
; CHECK-LABEL: call_alias_func:
; CHECK: .functype call_alias_func () -> ()
; CHECK-NEXT: call $push0=, alias_func
; CHECK-NEXT: drop $pop0{{$}}
; CHECK-NEXT: return{{$}}
  %call = call i32 @alias_func()
  ret void
}

define i8* @get_function_address() {
; CHECK-LABEL: get_function_address:
; CHECK:       global.get $push[[L0:[0-9]+]]=, bar@GOT{{$}}
; CHECK-NEXT:  return $pop[[L0]]{{$}}
; CHECK-NEXT:  end_function{{$}}

  ret i8* bitcast (i32 ()* @bar to i8*)
}

define i8* @get_function_address_hidden() {
; CHECK-LABEL: get_function_address_hidden:
; CHECK:       global.get $push[[L0:[0-9]+]]=, __table_base{{$}}
; CHECK-NEXT:  i32.const $push[[L1:[0-9]+]]=, hidden_function@TBREL{{$}}
; CHECK-NEXT:  i32.add $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT:  return $pop[[L2]]{{$}}
; CHECK-NEXT:  end_function{{$}}

  ret i8* bitcast (i32 ()* @hidden_function to i8*)
}
