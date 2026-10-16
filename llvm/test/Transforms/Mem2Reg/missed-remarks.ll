; RUN: opt -passes=mem2reg -pass-remarks-output=%t.yaml -disable-output < %s
; RUN: FileCheck %s -implicit-check-not=aggregate_deferred < %t.yaml

; mem2reg reports scalar allocas it cannot promote, naming the use that blocked
; promotion. Aggregates are not reported: promoting them is SROA's job, and
; SROA reports the ones it cannot split.

%struct.Pair = type { i32, i32 }

declare void @escape(ptr)

; CHECK:      Name:{{ +}}AllocaNotPromotable
; CHECK:      Function:{{ +}}escapes_to_call
; CHECK:      Address is passed to a call.

define i32 @escapes_to_call() {
entry:
  %escaping = alloca i32
  %promotable = alloca i32
  store i32 42, ptr %promotable
  call void @escape(ptr %escaping)
  %v = load i32, ptr %promotable
  ret i32 %v
}

; CHECK:      Name:{{ +}}AllocaNotPromotable
; CHECK:      Function:{{ +}}volatile_load
; CHECK:      Has a volatile load.

define i32 @volatile_load() {
entry:
  %x = alloca i32
  store i32 42, ptr %x
  %v = load volatile i32, ptr %x
  ret i32 %v
}

; CHECK:      Name:{{ +}}AllocaNotPromotable
; CHECK:      Function:{{ +}}address_stored
; CHECK:      Address is stored to memory.

define void @address_stored(ptr %sink) {
entry:
  %x = alloca i32
  store i32 42, ptr %x
  store ptr %x, ptr %sink
  ret void
}

; CHECK:      Name:{{ +}}AllocaNotPromotable
; CHECK:      Function:{{ +}}mixed_types
; CHECK:      Has loads and stores of different types.

define float @mixed_types() {
entry:
  %x = alloca i32
  store i32 42, ptr %x
  %v = load float, ptr %x
  ret float %v
}

; An aggregate mem2reg cannot see through: handed off to SROA, not reported.
define i32 @aggregate_deferred(i32 %n) {
entry:
  %p = alloca %struct.Pair
  %a = getelementptr inbounds %struct.Pair, ptr %p, i32 0, i32 0
  %b = getelementptr inbounds %struct.Pair, ptr %p, i32 0, i32 1
  store i32 %n, ptr %a
  store i32 %n, ptr %b
  %la = load i32, ptr %a
  %lb = load i32, ptr %b
  %sum = add i32 %la, %lb
  ret i32 %sum
}
