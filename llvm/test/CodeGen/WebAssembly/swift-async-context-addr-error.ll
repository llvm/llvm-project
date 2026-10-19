; RUN: not llc < %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc < %s -O0 -o /dev/null 2>&1 | FileCheck %s

; Local 0, which holds the async context, has no address.

target triple = "wasm32-unknown-unknown"

declare ptr @llvm.swift.async.context.addr()
declare void @escape(ptr)

; CHECK: in function escapes void (ptr): llvm.swift.async.context.addr can only be the address of loads and stores
define swiftcc void @escapes(ptr swiftasync %ctx) {
  %addr = call ptr @llvm.swift.async.context.addr()
  call void @escape(ptr %addr)
  ret void
}
