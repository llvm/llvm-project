; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s

target triple = "wasm32-unknown-unknown"

; CHECK: .weak f
define weak i32 @f() {
  unreachable
}

; CHECK: g:
; CHECK:  call h
define void @g() {
  tail call void @h( )
  ret void
}

; CHECK: bar:
; CHECK:   .int32 foo
; CHECK:   .size bar, 4
@bar = global ptr @foo

; CHECK: .weak h
declare extern_weak void @h()

; CHECK: .weak foo
@foo = extern_weak global i32
