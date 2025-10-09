; RUN: llc < %s -asm-verbose=false -wasm-keep-registers | FileCheck %s --check-prefix=REG
; RUN: llc < %s -asm-verbose=false | FileCheck %s

target triple = "wasm32-unknown-unknown"

; Test direct and indirect function call between mismatched signatures
; CHECK-LABEL: foo:
; CHECK-NEXT: .functype       foo (i32, i32, i32, i32) -> ()
define swiftcc void @foo(i32, i32) {
  ret void
}
@data = global ptr @foo

; CHECK-LABEL: bar:
; CHECK-NEXT: .functype       bar (i32, i32) -> ()
define swiftcc void @bar() {
  %1 = load ptr, ptr @data
; REG: call    foo, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}
  call swiftcc void @foo(i32 1, i32 2)

; REG: call_indirect   $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}
; CHECK: call_indirect __indirect_function_table, (i32, i32, i32, i32) -> ()
  call swiftcc void %1(i32 1, i32 2)

; REG: call_indirect   $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}
; CHECK: call_indirect __indirect_function_table, (i32, i32, i32, i32) -> ()
  call swiftcc void %1(i32 1, i32 2, i32 swiftself 3)

  %err = alloca swifterror ptr, align 4

; REG: call_indirect   $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}
; CHECK: call_indirect __indirect_function_table, (i32, i32, i32, i32) -> ()
  call swiftcc void %1(i32 1, i32 2, ptr swifterror %err)

; REG: call_indirect   $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}
; CHECK: call_indirect __indirect_function_table, (i32, i32, i32, i32) -> ()
  call swiftcc void %1(i32 1, i32 2, i32 swiftself 3, ptr swifterror %err)

  ret void
}
