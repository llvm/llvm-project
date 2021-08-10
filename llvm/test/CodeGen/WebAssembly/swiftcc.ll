; RUN: llc < %s -asm-verbose=false -wasm-keep-registers | FileCheck %s --check-prefix=REG
; RUN: llc < %s -asm-verbose=false | FileCheck %s

target triple = "wasm32-unknown-unknown"

; Test direct and indirect function call between mismatched signatures
; CHECK-LABEL: foo:
; CHECK-NEXT: .functype       foo (i32, i32, i32, i32) -> ()
define swiftcc void @foo(i32, i32) {
  ret void
}
@data = global i8* bitcast (void (i32, i32)* @foo to i8*)

; CHECK-LABEL: bar:
; CHECK-NEXT: .functype       bar (i32, i32) -> ()
define swiftcc void @bar() {
  %1 = load i8*, i8** @data
; REG: call    foo, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}
  call swiftcc void @foo(i32 1, i32 2)

  %2 = bitcast i8* %1 to void (i32, i32)*
; REG: call_indirect   $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}
; CHECK: call_indirect   (i32, i32, i32, i32) -> ()
  call swiftcc void %2(i32 1, i32 2)

  %3 = bitcast i8* %1 to void (i32, i32, i32)*
; REG: call_indirect   $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}
; CHECK: call_indirect   (i32, i32, i32, i32) -> ()
  call swiftcc void %3(i32 1, i32 2, i32 swiftself 3)

  %err = alloca swifterror i32*, align 4

  %4 = bitcast i8* %1 to void (i32, i32, i32**)*
; REG: call_indirect   $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}
; CHECK: call_indirect   (i32, i32, i32, i32) -> ()
  call swiftcc void %4(i32 1, i32 2, i32** swifterror %err)

  %5 = bitcast i8* %1 to void (i32, i32, i32, i32**)*
; REG: call_indirect   $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}
; CHECK: call_indirect   (i32, i32, i32, i32) -> ()
  call swiftcc void %5(i32 1, i32 2, i32 swiftself 3, i32** swifterror %err)

  ret void
}
