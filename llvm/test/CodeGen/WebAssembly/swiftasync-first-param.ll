; RUN: llc < %s -asm-verbose=false | FileCheck %s
; RUN: llc < %s -asm-verbose=false -O0 | FileCheck %s

; ABI contract: the async context must always be available in local 0, so the
; swiftasync argument is always parameter 0.

target triple = "wasm32-unknown-unknown"

; CHECK: .functype declared (i32, f64, i32, i32) -> ()

; CHECK-LABEL: callee:
; CHECK-NEXT: .functype callee (i32, i32, i64, i32, i32) -> (i64)
; CHECK-NEXT: local.get 0
; CHECK-NEXT: local.get 1
; CHECK-NEXT: i32.store 0
; CHECK-NEXT: local.get 2
define swiftcc i64 @callee(ptr %p, i64 %x, ptr swiftasync %ctx) {
  store ptr %p, ptr %ctx
  ret i64 %x
}

; CHECK-LABEL: caller:
; CHECK-NEXT: .functype caller (i32, i32, i32, i32) -> ()
; CHECK-NEXT: local.get 0
; CHECK-NEXT: local.get 1
; CHECK-NEXT: i64.const 7
; CHECK:      call callee
define swiftcc void @caller(ptr swiftasync %ctx, ptr %p) {
  %r = call swiftcc i64 @callee(ptr %p, i64 7, ptr swiftasync %ctx)
  ret void
}

; CHECK-LABEL: indirect_caller:
; CHECK-NEXT: .functype indirect_caller (i32, i32, i32, i32, i32) -> ()
; CHECK-NEXT: local.get 0
; CHECK-NEXT: local.get 1
; CHECK-NEXT: i64.const 7
; CHECK:      call_indirect __indirect_function_table, (i32, i32, i64, i32, i32) -> (i64)
define swiftcc void @indirect_caller(ptr swiftasync %ctx, ptr %p, ptr %fn) {
  %r = call swiftcc i64 %fn(ptr %p, i64 7, ptr swiftasync %ctx)
  ret void
}

; Multiple results are returned through a pointer parameter, which comes after
; the swiftasync argument.
; CHECK-LABEL: multiple_results:
; CHECK-NEXT: .functype multiple_results (i32, i32, i32, i32, i32) -> ()
; CHECK-NEXT: local.get 0
; CHECK-NEXT: local.get 2
; CHECK-NEXT: i32.store 0
; CHECK-NEXT: local.get 1
; CHECK-NEXT: i32.const 1
; CHECK-NEXT: i32.store 4
define swiftcc { i32, i32 } @multiple_results(i32 %x, ptr swiftasync %ctx) {
  store i32 %x, ptr %ctx
  %r0 = insertvalue { i32, i32 } poison, i32 %x, 0
  %r1 = insertvalue { i32, i32 } %r0, i32 1, 1
  ret { i32, i32 } %r1
}

; CHECK-LABEL: declared_caller:
; CHECK-NEXT: .functype declared_caller (i32, i32, i32) -> ()
; CHECK:      local.get 0
; CHECK-NEXT: f64.const 0x1p0
; CHECK:      call declared
declare swiftcc void @declared(double, ptr swiftasync)

define swiftcc void @declared_caller(ptr swiftasync %ctx) {
  call swiftcc void @declared(double 1.0, ptr swiftasync %ctx)
  ret void
}

; Only the callee has the swiftasync attribute.
; CHECK-LABEL: attribute_on_callee_only:
; CHECK-NEXT: .functype attribute_on_callee_only (i32) -> ()
; CHECK-NEXT: local.get 0
; CHECK-NEXT: i32.const 1
; CHECK-NEXT: call c_callee
declare void @c_callee(i32, ptr swiftasync)

define void @attribute_on_callee_only(ptr %p) {
  call void @c_callee(i32 1, ptr %p)
  ret void
}
