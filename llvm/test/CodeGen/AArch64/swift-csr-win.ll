; RUN: llc -mtriple aarch64-unknown-windows-msvc %s -o - | FileCheck %s

define internal swifttailcc void @"?future_adapter@@YWXPEAVAsyncContext@swift@@@Z"(ptr noundef swiftasync %_context) #0 {
entry:
  %add.ptr = getelementptr inbounds i8, ptr %_context, i64 -32
  %asyncEntryPoint = getelementptr inbounds i8, ptr %_context, i64 -24
  %0 = load ptr, ptr %asyncEntryPoint, align 8
  %closureContext = getelementptr inbounds i8, ptr %_context, i64 -16
  %1 = load ptr, ptr %closureContext, align 8
  %2 = load ptr, ptr %add.ptr, align 8
  musttail call swifttailcc void %0(ptr noundef %2, ptr noundef swiftasync %_context, ptr noundef swiftself %1) #17
  ret void
}

; Check that x20 isn't saved/restored at the prologue/epilogue which
; would interfere with the outgoing self parameter on x20 at the tail
; call.

; CHECK-NOT:  st{{.*}}x20
; CHECK:      ldp x1, x20, [x22, #-24]
; CHECK-NEXT: ldur x0, [x22, #-32]
; CHECK-NEXT: br x1
