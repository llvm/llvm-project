; RUN: llc < %s -asm-verbose=false -wasm-keep-registers | FileCheck %s

; Test that constant offsets can be folded into global addresses.

target triple = "wasm32-unknown-unknown"

@x = external dso_local global [0 x i32]
@y = dso_local global [50 x i32] zeroinitializer

; Test basic constant offsets of both defined and external symbols.

; CHECK-LABEL: test0:
; CHECK-NEXT: .functype test0 () -> (i32){{$}}
; CHECK-NEXT: i32.const $push0=, x+188{{$}}
; CHECK=NEXT: return $pop0{{$}}
define dso_local i32* @test0() {
  ret i32* getelementptr ([0 x i32], [0 x i32]* @x, i32 0, i32 47)
}

; CHECK-LABEL: test1:
; CHECK-NEXT: .functype test1 () -> (i32){{$}}
; CHECK-NEXT: i32.const $push0=, y+188{{$}}
; CHECK=NEXT: return $pop0{{$}}
define dso_local i32* @test1() {
  ret i32* getelementptr ([50 x i32], [50 x i32]* @y, i32 0, i32 47)
}

; Test zero offsets.

; CHECK-LABEL: test2:
; CHECK-NEXT: .functype test2 () -> (i32){{$}}
; CHECK-NEXT: i32.const $push0=, x{{$}}
; CHECK=NEXT: return $pop0{{$}}
define dso_local i32* @test2() {
  ret i32* getelementptr ([0 x i32], [0 x i32]* @x, i32 0, i32 0)
}

; CHECK-LABEL: test3:
; CHECK-NEXT: .functype test3 () -> (i32){{$}}
; CHECK-NEXT: i32.const $push0=, y{{$}}
; CHECK=NEXT: return $pop0{{$}}
define dso_local i32* @test3() {
  ret i32* getelementptr ([50 x i32], [50 x i32]* @y, i32 0, i32 0)
}

; Test negative offsets.

; CHECK-LABEL: test4:
; CHECK-NEXT: .functype test4 () -> (i32){{$}}
; CHECK-NEXT: i32.const $push0=, x-188{{$}}
; CHECK=NEXT: return $pop0{{$}}
define dso_local i32* @test4() {
  ret i32* getelementptr ([0 x i32], [0 x i32]* @x, i32 0, i32 -47)
}

; CHECK-LABEL: test5:
; CHECK-NEXT: .functype test5 () -> (i32){{$}}
; CHECK-NEXT: i32.const $push0=, y-188{{$}}
; CHECK=NEXT: return $pop0{{$}}
define dso_local i32* @test5() {
  ret i32* getelementptr ([50 x i32], [50 x i32]* @y, i32 0, i32 -47)
}
