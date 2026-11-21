; RUN: llc < %s -asm-verbose=false | FileCheck %s
; RUN: llc < %s -asm-verbose=false -O0 | FileCheck %s

; llvm.swift.async.context.addr is the address of local 0, which holds the
; async context. Await resume funclets receive the callee's context there and
; store their own through it before any call.

target triple = "wasm32-unknown-unknown"

declare swiftcc void @swift_task_dealloc(ptr)
declare ptr @llvm.swift.async.context.addr()

; CHECK-LABEL: resume:
; CHECK:      local.get 0
; CHECK-NEXT: i32.load 0
; CHECK-NOT:  call
; CHECK:      local.set 0{{$}}
; CHECK-NOT:  local.{{set|tee}} 0{{$}}
; CHECK:      end_function
define swiftcc void @resume(ptr swiftasync %callee_ctx, i32 %result) {
  %ctx = load ptr, ptr %callee_ctx
  %addr = call ptr @llvm.swift.async.context.addr()
  store ptr %ctx, ptr %addr
  %saved = getelementptr inbounds i8, ptr %ctx, i32 8
  %callee = load ptr, ptr %saved
  call swiftcc void @swift_task_dealloc(ptr %callee)
  %resume_addr = getelementptr inbounds i8, ptr %ctx, i32 4
  %resume = load ptr, ptr %resume_addr
  call swiftcc void %resume(ptr swiftasync %ctx, i32 %result)
  ret void
}

; The callee's context is read after the store, so it moves to another local.
; CHECK-LABEL: resume_reread:
; CHECK:      local.get 0
; CHECK-NEXT: local.set [[CALLEE:[0-9]+]]
; CHECK-NEXT: local.get [[CALLEE]]
; CHECK-NEXT: i32.load 0
; CHECK-NEXT: local.set 0
; CHECK-NEXT: local.get [[CALLEE]]
; CHECK-NEXT: i32.load 0
; CHECK-NOT:  local.{{set|tee}} 0{{$}}
; CHECK:      end_function
define swiftcc void @resume_reread(ptr swiftasync %callee_ctx, i32 %result) {
  %ctx = load ptr, ptr %callee_ctx
  %addr = call ptr @llvm.swift.async.context.addr()
  store ptr %ctx, ptr %addr
  %ctx2 = load ptr, ptr %callee_ctx
  call swiftcc void @swift_task_dealloc(ptr %callee_ctx)
  %resume_addr = getelementptr inbounds i8, ptr %ctx2, i32 4
  %resume = load ptr, ptr %resume_addr
  call swiftcc void %resume(ptr swiftasync %ctx2, i32 %result)
  ret void
}

; CHECK-LABEL: read:
; CHECK-NEXT: .functype read (i32, i32, i32) -> (i32)
; CHECK-NEXT: local.get 0
define swiftcc ptr @read(ptr swiftasync %ctx) {
  %addr = call ptr @llvm.swift.async.context.addr()
  %v = load ptr, ptr %addr
  ret ptr %v
}

; Without a swiftasync argument, there is no async context: the intrinsic is
; a stack slot.
; CHECK-LABEL: no_swiftasync:
; CHECK-NOT:  local.{{set|tee}} 0{{$}}
; CHECK:      i32.store
; CHECK-NOT:  local.{{set|tee}} 0{{$}}
; CHECK:      end_function
define swiftcc void @no_swiftasync(ptr %ctx) {
  %addr = call ptr @llvm.swift.async.context.addr()
  store ptr %ctx, ptr %addr
  ret void
}
