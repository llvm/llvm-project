; RUN: llc < %s -asm-verbose=false | FileCheck %s

; ABI contract: the async context must always be available in local 0, so no
; other value may be written to the local of the swiftasync argument, even after
; the last use of the argument.

target triple = "wasm32-unknown-unknown"

declare swiftcc void @swift_task_dealloc(ptr)

; Register coloring does not give local 0 to the parent context.
; CHECK-LABEL: swiftasync:
; CHECK-NEXT: .functype swiftasync (i32, i32, i32, i32) -> ()
; CHECK-NEXT: .local i32
; CHECK-NEXT: local.get 0
; CHECK-NEXT: i32.load 0
; CHECK-NEXT: local.tee 4
; CHECK-NOT:  local.{{set|tee}} 0{{$}}
; CHECK:      end_function
define swiftcc void @swiftasync(ptr swiftasync %callee_ctx, i32 %result) {
  %ctx = load ptr, ptr %callee_ctx
  %saved = getelementptr inbounds i8, ptr %ctx, i32 8
  %callee = load ptr, ptr %saved
  call swiftcc void @swift_task_dealloc(ptr %callee)
  %resume_addr = getelementptr inbounds i8, ptr %ctx, i32 4
  %resume = load ptr, ptr %resume_addr
  call swiftcc void %resume(ptr swiftasync %ctx, i32 %result)
  ret void
}

; Without swiftasync, local 0 is reused.
; CHECK-LABEL: no_swiftasync:
; CHECK-NEXT: .functype no_swiftasync (i32, i32, i32, i32) -> ()
; CHECK-NEXT: local.get 0
; CHECK-NEXT: i32.load 0
; CHECK-NEXT: local.tee 0
define swiftcc void @no_swiftasync(ptr %callee_ctx, i32 %result) {
  %ctx = load ptr, ptr %callee_ctx
  %saved = getelementptr inbounds i8, ptr %ctx, i32 8
  %callee = load ptr, ptr %saved
  call swiftcc void @swift_task_dealloc(ptr %callee)
  %resume_addr = getelementptr inbounds i8, ptr %ctx, i32 4
  %resume = load ptr, ptr %resume_addr
  call swiftcc void %resume(ptr %ctx, i32 %result)
  ret void
}

; Register coalescing merges the swiftasync argument with the loop variable,
; which then gets another local.
; CHECK-LABEL: coalesced:
; CHECK-NEXT: .functype coalesced (i32, i32, i32, i32) -> (i32)
; CHECK-NEXT: .local i32, i32
; CHECK-NEXT: local.get 0
; CHECK-NEXT: local.set 4
; CHECK-NOT:  local.{{set|tee}} 0{{$}}
; CHECK:      end_function
define swiftcc ptr @coalesced(ptr swiftasync %ctx, i32 %n) {
entry:
  br label %loop
loop:
  %p = phi ptr [ %ctx, %entry ], [ %next, %loop ]
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %next = load ptr, ptr %p
  %inc = add i32 %i, 1
  %c = icmp ult i32 %inc, %n
  br i1 %c, label %loop, label %exit
exit:
  ret ptr %next
}

; CHECK-LABEL: coalesced_with_call:
; CHECK-NEXT: .functype coalesced_with_call (i32, i32, i32, i32) -> (i32)
; CHECK:      local.get 0
; CHECK-NEXT: local.set [[P:[0-9]+]]
; CHECK-NOT:  local.{{set|tee}} 0{{$}}
; CHECK:      end_function
define swiftcc ptr @coalesced_with_call(ptr swiftasync %ctx, i32 %n) {
entry:
  %c0 = icmp eq i32 %n, 0
  br i1 %c0, label %exit, label %loop
loop:
  %p = phi ptr [ %ctx, %entry ], [ %next, %loop ]
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %next = load ptr, ptr %p
  call swiftcc void @swift_task_dealloc(ptr %next)
  %inc = add i32 %i, 1
  %c = icmp ult i32 %inc, %n
  br i1 %c, label %loop, label %exit
exit:
  %r = phi ptr [ %ctx, %entry ], [ %next, %loop ]
  ret ptr %r
}
