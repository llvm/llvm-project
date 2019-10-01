; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s -check-prefixes=CHECK,NON-PIC
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -relocation-model=pic | FileCheck %s -check-prefixes=CHECK,PIC


; Test folding constant offsets and symbols into load and store addresses under
; a variety of circumstances.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-emscripten"

@g = external global [0 x i32], align 4

; CHECK-LABEL: load_test0:
; CHECK-NEXT: .functype load_test0 () -> (i32){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 0{{$}}
; NON-PIC-NEXT:  i32.load  $push1=, g+40($pop0){{$}}
; PIC-NEXT:   global.get $push0=, g@GOT{{$}}
; PIC-NEXT:   i32.load  $push1=, 40($pop0){{$}}
; CHECK-NEXT: return    $pop1{{$}}
define i32 @load_test0() {
  %t = load i32, i32* getelementptr inbounds ([0 x i32], [0 x i32]* @g, i32 0, i32 10), align 4
  ret i32 %t
}

; CHECK-LABEL: load_test0_noinbounds:
; CHECK-NEXT: .functype load_test0_noinbounds () -> (i32){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 0{{$}}
; NON-PIC-NEXT:  i32.load  $push1=, g+40($pop0){{$}}
; PIC-NEXT:   global.get $push0=, g@GOT{{$}}
; PIC-NEXT:   i32.load  $push1=, 40($pop0){{$}}
; CHECK-NEXT: return    $pop1{{$}}
define i32 @load_test0_noinbounds() {
  %t = load i32, i32* getelementptr ([0 x i32], [0 x i32]* @g, i32 0, i32 10), align 4
  ret i32 %t
}

; TODO: load_test1 - load_test8 are disabled because folding GA+reg is disabled
; (there are cases where the value in the reg can be negative).
; Likewise for stores.

; CHECK-LABEL: load_test1:
; CHECK-NEXT: .functype load_test1 (i32) -> (i32){{$}}
; CHECK-NEX T: i32.const $push0=, 2{{$}}
; CHECK-NEX T: i32.shl   $push1=, $0, $pop0{{$}}
; CHECK-NEX T: i32.load  $push2=, g+40($pop1){{$}}
; CHECK-NEX T: return    $pop2{{$}}
define i32 @load_test1(i32 %n) {
  %add = add nsw i32 %n, 10
  %arrayidx = getelementptr inbounds [0 x i32], [0 x i32]* @g, i32 0, i32 %add
  %t = load i32, i32* %arrayidx, align 4
  ret i32 %t
}

; CHECK-LABEL: load_test2:
; CHECK-NEXT: .functype load_test2 (i32) -> (i32){{$}}
; CHECK-NEX T:  i32.const $push0=, 2{{$}}
; CHECK-NEX T: i32.shl   $push1=, $0, $pop0{{$}}
; CHECK-NEX T: i32.load  $push2=, g+40($pop1){{$}}
; CHECK-NEX T: return    $pop2{{$}}
define i32 @load_test2(i32 %n) {
  %add = add nsw i32 10, %n
  %arrayidx = getelementptr inbounds [0 x i32], [0 x i32]* @g, i32 0, i32 %add
  %t = load i32, i32* %arrayidx, align 4
  ret i32 %t
}

; CHECK-LABEL: load_test3:
; CHECK-NEXT: .functype load_test3 (i32) -> (i32){{$}}
; CHECK-NEX T: i32.const $push0=, 2{{$}}
; CHECK-NEX T: i32.shl   $push1=, $0, $pop0{{$}}
; CHECK-NEX T: i32.load  $push2=, g+40($pop1){{$}}
; CHECK-NEX T: return    $pop2{{$}}
define i32 @load_test3(i32 %n) {
  %add.ptr = getelementptr inbounds [0 x i32], [0 x i32]* @g, i32 0, i32 %n
  %add.ptr1 = getelementptr inbounds i32, i32* %add.ptr, i32 10
  %t = load i32, i32* %add.ptr1, align 4
  ret i32 %t
}

; CHECK-LABEL: load_test4:
; CHECK-NEXT: .functype load_test4 (i32) -> (i32){{$}}
; CHECK-NEX T: i32.const $push0=, 2{{$}}
; CHECK-NEX T: i32.shl   $push1=, $0, $pop0{{$}}
; CHECK-NEX T: i32.load  $push2=, g+40($pop1){{$}}
; CHECK-NEX T: return    $pop2{{$}}
define i32 @load_test4(i32 %n) {
  %add.ptr = getelementptr inbounds i32, i32* getelementptr inbounds ([0 x i32], [0 x i32]* @g, i32 0, i32 10), i32 %n
  %t = load i32, i32* %add.ptr, align 4
  ret i32 %t
}

; CHECK-LABEL: load_test5:
; CHECK-NEXT: .functype load_test5 (i32) -> (i32){{$}}
; CHECK-NEX T: i32.const $push0=, 2{{$}}
; CHECK-NEX T: i32.shl   $push1=, $0, $pop0{{$}}
; CHECK-NEX T: i32.load  $push2=, g+40($pop1){{$}}
; CHECK-NEX T: return    $pop2{{$}}
define i32 @load_test5(i32 %n) {
  %add.ptr = getelementptr inbounds i32, i32* getelementptr inbounds ([0 x i32], [0 x i32]* @g, i32 0, i32 10), i32 %n
  %t = load i32, i32* %add.ptr, align 4
  ret i32 %t
}

; CHECK-LABEL: load_test6:
; CHECK-NEXT: .functype load_test6 (i32) -> (i32){{$}}
; CHECK-NEX T:  i32.const $push0=, 2{{$}}
; CHECK-NEX T: i32.shl   $push1=, $0, $pop0{{$}}
; CHECK-NEX T: i32.load  $push2=, g+40($pop1){{$}}
; CHECK-NEX T: return    $pop2{{$}}
define i32 @load_test6(i32 %n) {
  %add = add nsw i32 %n, 10
  %add.ptr = getelementptr inbounds [0 x i32], [0 x i32]* @g, i32 0, i32 %add
  %t = load i32, i32* %add.ptr, align 4
  ret i32 %t
}

; CHECK-LABEL: load_test7:
; CHECK-NEXT: .functype load_test7 (i32) -> (i32){{$}}
; CHECK-NEX T: i32.const $push0=, 2{{$}}
; CHECK-NEX T: i32.shl   $push1=, $0, $pop0{{$}}
; CHECK-NEX T: i32.load  $push2=, g+40($pop1){{$}}
; CHECK-NEX T: return    $pop2{{$}}
define i32 @load_test7(i32 %n) {
  %add.ptr = getelementptr inbounds [0 x i32], [0 x i32]* @g, i32 0, i32 %n
  %add.ptr1 = getelementptr inbounds i32, i32* %add.ptr, i32 10
  %t = load i32, i32* %add.ptr1, align 4
  ret i32 %t
}

; CHECK-LABEL: load_test8:
; CHECK-NEXT: .functype load_test8 (i32) -> (i32){{$}}
; CHECK-NEX T: i32.const $push0=, 2{{$}}
; CHECK-NEX T: i32.shl   $push1=, $0, $pop0{{$}}
; CHECK-NEX T: i32.load  $push2=, g+40($pop1){{$}}
; CHECK-NEX T: return    $pop2{{$}}
define i32 @load_test8(i32 %n) {
  %add = add nsw i32 10, %n
  %add.ptr = getelementptr inbounds [0 x i32], [0 x i32]* @g, i32 0, i32 %add
  %t = load i32, i32* %add.ptr, align 4
  ret i32 %t
}

; CHECK-LABEL: load_test9:
; CHECK-NEXT:  .functype load_test9 () -> (i32){{$}}
; NON-PIC-NEXT: i32.const  $push0=, 0{{$}}
; NON-PIC-NEXT: i32.load   $push1=, g-40($pop0){{$}}
; NON-PIC_NEXT: return     $pop1{{$}}

; PIC-NEXT: global.get $push1=, g@GOT{{$}}
; PIC-NEXT: i32.const  $push0=, -40{{$}}
; PIC-NEXT: i32.add    $push2=, $pop1, $pop0{{$}}
; PIC-NEXT: i32.load   $push3=, 0($pop2)
; PIC-NEXT: return     $pop3{{$}}
define i32 @load_test9() {
  %t = load i32, i32* getelementptr inbounds ([0 x i32], [0 x i32]* @g, i32 0, i32 1073741814), align 4
  ret i32 %t
}

; CHECK-LABEL: load_test10:
; CHECK-NEXT: .functype load_test10 (i32) -> (i32){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 2{{$}}
; NON-PIC-NEXT:  i32.shl   $push1=, $0, $pop0{{$}}
; NON-PIC-NEXT:  i32.const $push2=, g-40{{$}}
; NON-PIC-NEXT:  i32.add   $push3=, $pop1, $pop2{{$}}
; NON-PIC-NEXT:  i32.load  $push4=, 0($pop3){{$}}
; NON-PIC-NEXT:  return    $pop4{{$}}

; PIC-NEXT:   i32.const $push0=, 2{{$}}
; PIC-NEXT:   i32.shl   $push1=, $0, $pop0{{$}}
; PIC-NEXT:   global.get $push2=, g@GOT{{$}}
; PIC-NEXT:   i32.add   $push3=, $pop1, $pop2{{$}}
; PIC-NEXT:   i32.const $push4=, -40{{$}}
; PIC-NEXT:   i32.add   $push5=, $pop3, $pop4{{$}}
; PIC-NEXT:   i32.load  $push6=, 0($pop5){{$}}
; PIC-NEXT:   return    $pop6{{$}}
define i32 @load_test10(i32 %n) {
  %add = add nsw i32 %n, -10
  %arrayidx = getelementptr inbounds [0 x i32], [0 x i32]* @g, i32 0, i32 %add
  %t = load i32, i32* %arrayidx, align 4
  ret i32 %t
}

; CHECK-LABEL: load_test11:
; CHECK-NEXT: .functype load_test11 (i32) -> (i32){{$}}
; CHECK-NEXT: i32.load  $push0=, 40($0){{$}}
; CHECK-NEXT: return    $pop0{{$}}
define i32 @load_test11(i32* %p) {
  %arrayidx = getelementptr inbounds i32, i32* %p, i32 10
  %t = load i32, i32* %arrayidx, align 4
  ret i32 %t
}

; CHECK-LABEL: load_test11_noinbounds:
; CHECK-NEXT: .functype load_test11_noinbounds (i32) -> (i32){{$}}
; CHECK-NEXT: i32.const $push0=, 40{{$}}
; CHECK-NEXT: i32.add   $push1=, $0, $pop0{{$}}
; CHECK-NEXT: i32.load  $push2=, 0($pop1){{$}}
; CHECK-NEXT: return    $pop2{{$}}
define i32 @load_test11_noinbounds(i32* %p) {
  %arrayidx = getelementptr i32, i32* %p, i32 10
  %t = load i32, i32* %arrayidx, align 4
  ret i32 %t
}

; CHECK-LABEL: load_test12:
; CHECK-NEXT: .functype load_test12 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: i32.const $push0=, 2{{$}}
; CHECK-NEXT: i32.shl   $push1=, $1, $pop0{{$}}
; CHECK-NEXT: i32.add   $push2=, $pop1, $0{{$}}
; CHECK-NEXT: i32.const $push3=, 40{{$}}
; CHECK-NEXT: i32.add   $push4=, $pop2, $pop3{{$}}
; CHECK-NEXT: i32.load  $push5=, 0($pop4){{$}}
; CHECK-NEXT: return    $pop5{{$}}
define i32 @load_test12(i32* %p, i32 %n) {
  %add = add nsw i32 %n, 10
  %arrayidx = getelementptr inbounds i32, i32* %p, i32 %add
  %t = load i32, i32* %arrayidx, align 4
  ret i32 %t
}

; CHECK-LABEL: load_test13:
; CHECK-NEXT: .functype load_test13 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: i32.const $push0=, 2{{$}}
; CHECK-NEXT: i32.shl   $push1=, $1, $pop0{{$}}
; CHECK-NEXT: i32.add   $push2=, $pop1, $0{{$}}
; CHECK-NEXT: i32.const $push3=, 40{{$}}
; CHECK-NEXT: i32.add   $push4=, $pop2, $pop3{{$}}
; CHECK-NEXT: i32.load  $push5=, 0($pop4){{$}}
; CHECK-NEXT: return    $pop5{{$}}
define i32 @load_test13(i32* %p, i32 %n) {
  %add = add nsw i32 10, %n
  %arrayidx = getelementptr inbounds i32, i32* %p, i32 %add
  %t = load i32, i32* %arrayidx, align 4
  ret i32 %t
}

; CHECK-LABEL: load_test14:
; CHECK-NEXT: .functype load_test14 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: i32.const $push0=, 2{{$}}
; CHECK-NEXT: i32.shl   $push1=, $1, $pop0{{$}}
; CHECK-NEXT: i32.add   $push2=, $0, $pop1{{$}}
; CHECK-NEXT: i32.load  $push3=, 40($pop2){{$}}
; CHECK-NEXT: return    $pop3{{$}}
define i32 @load_test14(i32* %p, i32 %n) {
  %add.ptr = getelementptr inbounds i32, i32* %p, i32 %n
  %add.ptr1 = getelementptr inbounds i32, i32* %add.ptr, i32 10
  %t = load i32, i32* %add.ptr1, align 4
  ret i32 %t
}

; CHECK-LABEL: load_test15:
; CHECK-NEXT: .functype load_test15 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: i32.const $push0=, 2{{$}}
; CHECK-NEXT: i32.shl   $push1=, $1, $pop0{{$}}
; CHECK-NEXT: i32.add   $push2=, $0, $pop1{{$}}
; CHECK-NEXT: i32.const $push3=, 40{{$}}
; CHECK-NEXT: i32.add   $push4=, $pop2, $pop3{{$}}
; CHECK-NEXT: i32.load  $push5=, 0($pop4){{$}}
; CHECK-NEXT: return    $pop5{{$}}
define i32 @load_test15(i32* %p, i32 %n) {
  %add.ptr = getelementptr inbounds i32, i32* %p, i32 10
  %add.ptr1 = getelementptr inbounds i32, i32* %add.ptr, i32 %n
  %t = load i32, i32* %add.ptr1, align 4
  ret i32 %t
}

; CHECK-LABEL: load_test16:
; CHECK-NEXT: .functype load_test16 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: i32.const $push0=, 2{{$}}
; CHECK-NEXT: i32.shl   $push1=, $1, $pop0{{$}}
; CHECK-NEXT: i32.add   $push2=, $0, $pop1{{$}}
; CHECK-NEXT: i32.const $push3=, 40{{$}}
; CHECK-NEXT: i32.add   $push4=, $pop2, $pop3{{$}}
; CHECK-NEXT: i32.load  $push5=, 0($pop4){{$}}
; CHECK-NEXT: return    $pop5{{$}}
define i32 @load_test16(i32* %p, i32 %n) {
  %add.ptr = getelementptr inbounds i32, i32* %p, i32 10
  %add.ptr1 = getelementptr inbounds i32, i32* %add.ptr, i32 %n
  %t = load i32, i32* %add.ptr1, align 4
  ret i32 %t
}

; CHECK-LABEL: load_test17:
; CHECK-NEXT: .functype load_test17 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: i32.const $push0=, 2{{$}}
; CHECK-NEXT: i32.shl   $push1=, $1, $pop0{{$}}
; CHECK-NEXT: i32.add   $push2=, $pop1, $0{{$}}
; CHECK-NEXT: i32.const $push3=, 40{{$}}
; CHECK-NEXT: i32.add   $push4=, $pop2, $pop3{{$}}
; CHECK-NEXT: i32.load  $push5=, 0($pop4){{$}}
; CHECK-NEXT: return    $pop5{{$}}
define i32 @load_test17(i32* %p, i32 %n) {
  %add = add nsw i32 %n, 10
  %add.ptr = getelementptr inbounds i32, i32* %p, i32 %add
  %t = load i32, i32* %add.ptr, align 4
  ret i32 %t
}

; CHECK-LABEL: load_test18:
; CHECK-NEXT: .functype load_test18 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: i32.const $push0=, 2{{$}}
; CHECK-NEXT: i32.shl   $push1=, $1, $pop0{{$}}
; CHECK-NEXT: i32.add   $push2=, $0, $pop1{{$}}
; CHECK-NEXT: i32.load  $push3=, 40($pop2){{$}}
; CHECK-NEXT: return    $pop3{{$}}
define i32 @load_test18(i32* %p, i32 %n) {
  %add.ptr = getelementptr inbounds i32, i32* %p, i32 %n
  %add.ptr1 = getelementptr inbounds i32, i32* %add.ptr, i32 10
  %t = load i32, i32* %add.ptr1, align 4
  ret i32 %t
}

; CHECK-LABEL: load_test19:
; CHECK-NEXT: .functype load_test19 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: i32.const $push0=, 2{{$}}
; CHECK-NEXT: i32.shl   $push1=, $1, $pop0{{$}}
; CHECK-NEXT: i32.add   $push2=, $pop1, $0{{$}}
; CHECK-NEXT: i32.const $push3=, 40{{$}}
; CHECK-NEXT: i32.add   $push4=, $pop2, $pop3{{$}}
; CHECK-NEXT: i32.load  $push5=, 0($pop4){{$}}
; CHECK-NEXT: return    $pop5{{$}}
define i32 @load_test19(i32* %p, i32 %n) {
  %add = add nsw i32 10, %n
  %add.ptr = getelementptr inbounds i32, i32* %p, i32 %add
  %t = load i32, i32* %add.ptr, align 4
  ret i32 %t
}

; CHECK-LABEL: load_test20:
; CHECK-NEXT: .functype load_test20 (i32) -> (i32){{$}}
; CHECK-NEXT: i32.const $push0=, -40{{$}}
; CHECK-NEXT: i32.add   $push1=, $0, $pop0{{$}}
; CHECK-NEXT: i32.load  $push2=, 0($pop1){{$}}
; CHECK-NEXT: return    $pop2{{$}}
define i32 @load_test20(i32* %p) {
  %arrayidx = getelementptr inbounds i32, i32* %p, i32 -10
  %t = load i32, i32* %arrayidx, align 4
  ret i32 %t
}

; CHECK-LABEL: load_test21:
; CHECK-NEXT: .functype load_test21 (i32, i32) -> (i32){{$}}
; CHECK-NEXT: i32.const $push0=, 2{{$}}
; CHECK-NEXT: i32.shl   $push1=, $1, $pop0{{$}}
; CHECK-NEXT: i32.add   $push2=, $pop1, $0{{$}}
; CHECK-NEXT: i32.const $push3=, -40{{$}}
; CHECK-NEXT: i32.add   $push4=, $pop2, $pop3{{$}}
; CHECK-NEXT: i32.load  $push5=, 0($pop4){{$}}
; CHECK-NEXT: return    $pop5{{$}}
define i32 @load_test21(i32* %p, i32 %n) {
  %add = add nsw i32 %n, -10
  %arrayidx = getelementptr inbounds i32, i32* %p, i32 %add
  %t = load i32, i32* %arrayidx, align 4
  ret i32 %t
}

; CHECK-LABEL: store_test0:
; CHECK-NEXT: .functype store_test0 (i32) -> (){{$}}
; NON-PIC-NEXT: i32.const $push0=, 0{{$}}
; NON-PIC-NEXT: i32.store g+40($pop0), $0{{$}}
; PIC-NEXT:     global.get $push0=, g@GOT{{$}}
; PIC-NEXT:     i32.store 40($pop0), $0
; CHECK-NEXT:   return{{$}}
define void @store_test0(i32 %i) {
  store i32 %i, i32* getelementptr inbounds ([0 x i32], [0 x i32]* @g, i32 0, i32 10), align 4
  ret void
}

; CHECK-LABEL: store_test0_noinbounds:
; CHECK-NEXT: .functype store_test0_noinbounds (i32) -> (){{$}}
; NON-PIC-NEXT: i32.const $push0=, 0{{$}}
; NON-PIC-NEXT: i32.store g+40($pop0), $0{{$}}
; PIC-NEXT:     global.get $push0=, g@GOT{{$}}
; PIC-NEXT:     i32.store 40($pop0), $0{{$}}
; CHECK-NEXT:  return{{$}}
define void @store_test0_noinbounds(i32 %i) {
  store i32 %i, i32* getelementptr ([0 x i32], [0 x i32]* @g, i32 0, i32 10), align 4
  ret void
}

; CHECK-LABEL: store_test1:
; CHECK-NEXT: .functype store_test1 (i32, i32) -> (){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 2{{$}}
; NON-PIC-NEXT:  i32.shl   $push1=, $0, $pop0{{$}}
; CHECK-NEX T: i32.store g+40($pop1), $1{{$}}
; CHECK-NEX T: return{{$}}
define void @store_test1(i32 %n, i32 %i) {
  %add = add nsw i32 %n, 10
  %arrayidx = getelementptr inbounds [0 x i32], [0 x i32]* @g, i32 0, i32 %add
  store i32 %i, i32* %arrayidx, align 4
  ret void
}

; CHECK-LABEL: store_test2:
; CHECK-NEXT: .functype store_test2 (i32, i32) -> (){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 2{{$}}
; NON-PIC-NEXT:  i32.shl   $push1=, $0, $pop0{{$}}
; CHECK-NEX T: i32.store g+40($pop1), $1{{$}}
; CHECK-NEX T: return{{$}}
define void @store_test2(i32 %n, i32 %i) {
  %add = add nsw i32 10, %n
  %arrayidx = getelementptr inbounds [0 x i32], [0 x i32]* @g, i32 0, i32 %add
  store i32 %i, i32* %arrayidx, align 4
  ret void
}

; CHECK-LABEL: store_test3:
; CHECK-NEXT: .functype store_test3 (i32, i32) -> (){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 2{{$}}
; NON-PIC-NEXT:  i32.shl   $push1=, $0, $pop0{{$}}
; CHECK-NEX T: i32.store g+40($pop1), $1{{$}}
; CHECK-NEX T: return{{$}}
define void @store_test3(i32 %n, i32 %i) {
  %add.ptr = getelementptr inbounds [0 x i32], [0 x i32]* @g, i32 0, i32 %n
  %add.ptr1 = getelementptr inbounds i32, i32* %add.ptr, i32 10
  store i32 %i, i32* %add.ptr1, align 4
  ret void
}

; CHECK-LABEL: store_test4:
; CHECK-NEXT: .functype store_test4 (i32, i32) -> (){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 2{{$}}
; NON-PIC-NEXT:  i32.shl   $push1=, $0, $pop0{{$}}
; CHECK-NEX T: i32.store g+40($pop1), $1{{$}}
; CHECK-NEX T: return{{$}}
define void @store_test4(i32 %n, i32 %i) {
  %add.ptr = getelementptr inbounds i32, i32* getelementptr inbounds ([0 x i32], [0 x i32]* @g, i32 0, i32 10), i32 %n
  store i32 %i, i32* %add.ptr, align 4
  ret void
}

; CHECK-LABEL: store_test5:
; CHECK-NEXT: .functype store_test5 (i32, i32) -> (){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 2{{$}}
; NON-PIC-NEXT:  i32.shl   $push1=, $0, $pop0{{$}}
; CHECK-NEX T: i32.store g+40($pop1), $1{{$}}
; CHECK-NEX T: return{{$}}
define void @store_test5(i32 %n, i32 %i) {
  %add.ptr = getelementptr inbounds i32, i32* getelementptr inbounds ([0 x i32], [0 x i32]* @g, i32 0, i32 10), i32 %n
  store i32 %i, i32* %add.ptr, align 4
  ret void
}

; CHECK-LABEL: store_test6:
; CHECK-NEXT: .functype store_test6 (i32, i32) -> (){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 2{{$}}
; NON-PIC-NEXT:  i32.shl   $push1=, $0, $pop0{{$}}
; CHECK-NEX T: i32.store g+40($pop1), $1{{$}}
; CHECK-NEX T: return{{$}}
define void @store_test6(i32 %n, i32 %i) {
  %add = add nsw i32 %n, 10
  %add.ptr = getelementptr inbounds [0 x i32], [0 x i32]* @g, i32 0, i32 %add
  store i32 %i, i32* %add.ptr, align 4
  ret void
}

; CHECK-LABEL: store_test7:
; CHECK-NEXT: .functype store_test7 (i32, i32) -> (){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 2{{$}}
; NON-PIC-NEXT:  i32.shl   $push1=, $0, $pop0{{$}}
; CHECK-NEX T: i32.store g+40($pop1), $1{{$}}
; CHECK-NEX T: return{{$}}
define void @store_test7(i32 %n, i32 %i) {
  %add.ptr = getelementptr inbounds [0 x i32], [0 x i32]* @g, i32 0, i32 %n
  %add.ptr1 = getelementptr inbounds i32, i32* %add.ptr, i32 10
  store i32 %i, i32* %add.ptr1, align 4
  ret void
}

; CHECK-LABEL: store_test8:
; CHECK-NEXT: .functype store_test8 (i32, i32) -> (){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 2{{$}}
; NON-PIC-NEXT:  i32.shl   $push1=, $0, $pop0{{$}}
; CHECK-NEX T: i32.store g+40($pop1), $1{{$}}
; CHECK-NEX T: return{{$}}
define void @store_test8(i32 %n, i32 %i) {
  %add = add nsw i32 10, %n
  %add.ptr = getelementptr inbounds [0 x i32], [0 x i32]* @g, i32 0, i32 %add
  store i32 %i, i32* %add.ptr, align 4
  ret void
}

; CHECK-LABEL: store_test9:
; CHECK-NEXT: .functype store_test9 (i32) -> (){{$}}
; NON-PIC-NEXT:  i32.const  $push0=, 0{{$}}
; NON-PIC-NEXT:  i32.store  g-40($pop0), $0{{$}}
; PIC-NEXT:      global.get $push1=, g@GOT{{$}}
; PIC-NEXT:      i32.const  $push0=, -40{{$}}
; PIC-NEXT:      i32.add    $push2=, $pop1, $pop0{{$}}
; PIC-NEXT:      i32.store  0($pop2), $0
; CHECK-NEXT:  return{{$}}
define void @store_test9(i32 %i) {
  store i32 %i, i32* getelementptr inbounds ([0 x i32], [0 x i32]* @g, i32 0, i32 1073741814), align 4
  ret void
}

; CHECK-LABEL: store_test10:
; CHECK-NEXT: .functype store_test10 (i32, i32) -> (){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 2{{$}}
; NON-PIC-NEXT:  i32.shl   $push1=, $0, $pop0{{$}}
; NON-PIC-NEXT:  i32.const $push2=, g-40{{$}}
; NON-PIC-NEXT:  i32.add   $push3=, $pop1, $pop2{{$}}
; NON-PIC-NEXT:  i32.store 0($pop3), $1{{$}}
; PIC-NEXT: i32.const  $push0=, 2{{$}}
; PIC-NEXT: i32.shl    $push1=, $0, $pop0{{$}}
; PIC-NEXT: global.get $push2=, g@GOT{{$}}
; PIC-NEXT: i32.add    $push3=, $pop1, $pop2{{$}}
; PIC-NEXT: i32.const  $push4=, -40{{$}}
; PIC-NEXT: i32.add    $push5=, $pop3, $pop4{{$}}
; PIC-NEXT: i32.store  0($pop5), $1{{$}}
; CHECK-NEXT:  return{{$}}
define void @store_test10(i32 %n, i32 %i) {
  %add = add nsw i32 %n, -10
  %arrayidx = getelementptr inbounds [0 x i32], [0 x i32]* @g, i32 0, i32 %add
  store i32 %i, i32* %arrayidx, align 4
  ret void
}

; CHECK-LABEL: store_test11:
; CHECK-NEXT: .functype store_test11 (i32, i32) -> (){{$}}
; CHECK-NEXT:  i32.store 40($0), $1{{$}}
; CHECK-NEXT:  return{{$}}
define void @store_test11(i32* %p, i32 %i) {
  %arrayidx = getelementptr inbounds i32, i32* %p, i32 10
  store i32 %i, i32* %arrayidx, align 4
  ret void
}

; CHECK-LABEL: store_test11_noinbounds:
; CHECK-NEXT: .functype store_test11_noinbounds (i32, i32) -> (){{$}}
; CHECK-NEXT:  i32.const $push0=, 40{{$}}
; CHECK-NEXT:  i32.add   $push1=, $0, $pop0{{$}}
; CHECK-NEXT:  i32.store 0($pop1), $1{{$}}
; CHECK-NEXT:  return{{$}}
define void @store_test11_noinbounds(i32* %p, i32 %i) {
  %arrayidx = getelementptr i32, i32* %p, i32 10
  store i32 %i, i32* %arrayidx, align 4
  ret void
}

; CHECK-LABEL: store_test12:
; CHECK-NEXT: .functype store_test12 (i32, i32, i32) -> (){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 2{{$}}
; NON-PIC-NEXT:  i32.shl   $push1=, $1, $pop0{{$}}
; NON-PIC-NEXT:  i32.add   $push2=, $pop1, $0{{$}}
; NON-PIC-NEXT:  i32.const $push3=, 40{{$}}
; NON-PIC-NEXT:  i32.add   $push4=, $pop2, $pop3{{$}}
; NON-PIC-NEXT:  i32.store 0($pop4), $2{{$}}
; NON-PIC-NEXT:  return{{$}}
define void @store_test12(i32* %p, i32 %n, i32 %i) {
  %add = add nsw i32 %n, 10
  %arrayidx = getelementptr inbounds i32, i32* %p, i32 %add
  store i32 %i, i32* %arrayidx, align 4
  ret void
}

; CHECK-LABEL: store_test13:
; CHECK-NEXT: .functype store_test13 (i32, i32, i32) -> (){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 2{{$}}
; NON-PIC-NEXT:  i32.shl   $push1=, $1, $pop0{{$}}
; NON-PIC-NEXT:  i32.add   $push2=, $pop1, $0{{$}}
; NON-PIC-NEXT:  i32.const $push3=, 40{{$}}
; NON-PIC-NEXT:  i32.add   $push4=, $pop2, $pop3{{$}}
; NON-PIC-NEXT:  i32.store 0($pop4), $2{{$}}
; NON-PIC-NEXT:  return{{$}}
define void @store_test13(i32* %p, i32 %n, i32 %i) {
  %add = add nsw i32 10, %n
  %arrayidx = getelementptr inbounds i32, i32* %p, i32 %add
  store i32 %i, i32* %arrayidx, align 4
  ret void
}

; CHECK-LABEL: store_test14:
; CHECK-NEXT: .functype store_test14 (i32, i32, i32) -> (){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 2{{$}}
; NON-PIC-NEXT:  i32.shl   $push1=, $1, $pop0{{$}}
; NON-PIC-NEXT:  i32.add   $push2=, $0, $pop1{{$}}
; NON-PIC-NEXT:  i32.store 40($pop2), $2{{$}}
; NON-PIC-NEXT:  return{{$}}
define void @store_test14(i32* %p, i32 %n, i32 %i) {
  %add.ptr = getelementptr inbounds i32, i32* %p, i32 %n
  %add.ptr1 = getelementptr inbounds i32, i32* %add.ptr, i32 10
  store i32 %i, i32* %add.ptr1, align 4
  ret void
}

; CHECK-LABEL: store_test15:
; CHECK-NEXT: .functype store_test15 (i32, i32, i32) -> (){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 2{{$}}
; NON-PIC-NEXT:  i32.shl   $push1=, $1, $pop0{{$}}
; NON-PIC-NEXT:  i32.add   $push2=, $0, $pop1{{$}}
; NON-PIC-NEXT:  i32.const $push3=, 40{{$}}
; NON-PIC-NEXT:  i32.add   $push4=, $pop2, $pop3{{$}}
; NON-PIC-NEXT:  i32.store 0($pop4), $2{{$}}
; NON-PIC-NEXT:  return{{$}}
define void @store_test15(i32* %p, i32 %n, i32 %i) {
  %add.ptr = getelementptr inbounds i32, i32* %p, i32 10
  %add.ptr1 = getelementptr inbounds i32, i32* %add.ptr, i32 %n
  store i32 %i, i32* %add.ptr1, align 4
  ret void
}

; CHECK-LABEL: store_test16:
; CHECK-NEXT: .functype store_test16 (i32, i32, i32) -> (){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 2{{$}}
; NON-PIC-NEXT:  i32.shl   $push1=, $1, $pop0{{$}}
; NON-PIC-NEXT:  i32.add   $push2=, $0, $pop1{{$}}
; NON-PIC-NEXT:  i32.const $push3=, 40{{$}}
; NON-PIC-NEXT:  i32.add   $push4=, $pop2, $pop3{{$}}
; NON-PIC-NEXT:  i32.store 0($pop4), $2{{$}}
; NON-PIC-NEXT:  return{{$}}
define void @store_test16(i32* %p, i32 %n, i32 %i) {
  %add.ptr = getelementptr inbounds i32, i32* %p, i32 10
  %add.ptr1 = getelementptr inbounds i32, i32* %add.ptr, i32 %n
  store i32 %i, i32* %add.ptr1, align 4
  ret void
}

; CHECK-LABEL: store_test17:
; CHECK-NEXT: .functype store_test17 (i32, i32, i32) -> (){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 2{{$}}
; NON-PIC-NEXT:  i32.shl   $push1=, $1, $pop0{{$}}
; NON-PIC-NEXT:  i32.add   $push2=, $pop1, $0{{$}}
; NON-PIC-NEXT:  i32.const $push3=, 40{{$}}
; NON-PIC-NEXT:  i32.add   $push4=, $pop2, $pop3{{$}}
; NON-PIC-NEXT:  i32.store 0($pop4), $2{{$}}
; NON-PIC-NEXT:  return{{$}}
define void @store_test17(i32* %p, i32 %n, i32 %i) {
  %add = add nsw i32 %n, 10
  %add.ptr = getelementptr inbounds i32, i32* %p, i32 %add
  store i32 %i, i32* %add.ptr, align 4
  ret void
}

; CHECK-LABEL: store_test18:
; CHECK-NEXT: .functype store_test18 (i32, i32, i32) -> (){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 2{{$}}
; NON-PIC-NEXT:  i32.shl   $push1=, $1, $pop0{{$}}
; NON-PIC-NEXT:  i32.add   $push2=, $0, $pop1{{$}}
; NON-PIC-NEXT:  i32.store 40($pop2), $2{{$}}
; NON-PIC-NEXT:  return{{$}}
define void @store_test18(i32* %p, i32 %n, i32 %i) {
  %add.ptr = getelementptr inbounds i32, i32* %p, i32 %n
  %add.ptr1 = getelementptr inbounds i32, i32* %add.ptr, i32 10
  store i32 %i, i32* %add.ptr1, align 4
  ret void
}

; CHECK-LABEL: store_test19:
; CHECK-NEXT: .functype store_test19 (i32, i32, i32) -> (){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 2{{$}}
; NON-PIC-NEXT:  i32.shl   $push1=, $1, $pop0{{$}}
; NON-PIC-NEXT:  i32.add   $push2=, $pop1, $0{{$}}
; NON-PIC-NEXT:  i32.const $push3=, 40{{$}}
; NON-PIC-NEXT:  i32.add   $push4=, $pop2, $pop3{{$}}
; NON-PIC-NEXT:  i32.store 0($pop4), $2{{$}}
; NON-PIC-NEXT:  return{{$}}
define void @store_test19(i32* %p, i32 %n, i32 %i) {
  %add = add nsw i32 10, %n
  %add.ptr = getelementptr inbounds i32, i32* %p, i32 %add
  store i32 %i, i32* %add.ptr, align 4
  ret void
}

; CHECK-LABEL: store_test20:
; CHECK-NEXT: .functype store_test20 (i32, i32) -> (){{$}}
; NON-PIC-NEXT:  i32.const $push0=, -40{{$}}
; NON-PIC-NEXT:  i32.add   $push1=, $0, $pop0{{$}}
; NON-PIC-NEXT:  i32.store 0($pop1), $1{{$}}
; NON-PIC-NEXT:  return{{$}}
define void @store_test20(i32* %p, i32 %i) {
  %arrayidx = getelementptr inbounds i32, i32* %p, i32 -10
  store i32 %i, i32* %arrayidx, align 4
  ret void
}

; CHECK-LABEL: store_test21:
; CHECK-NEXT: .functype store_test21 (i32, i32, i32) -> (){{$}}
; NON-PIC-NEXT:  i32.const $push0=, 2{{$}}
; NON-PIC-NEXT:  i32.shl   $push1=, $1, $pop0{{$}}
; NON-PIC-NEXT:  i32.add   $push2=, $pop1, $0{{$}}
; NON-PIC-NEXT:  i32.const $push3=, -40{{$}}
; NON-PIC-NEXT:  i32.add   $push4=, $pop2, $pop3{{$}}
; NON-PIC-NEXT:  i32.store 0($pop4), $2{{$}}
; NON-PIC-NEXT:  return{{$}}
define void @store_test21(i32* %p, i32 %n, i32 %i) {
  %add = add nsw i32 %n, -10
  %arrayidx = getelementptr inbounds i32, i32* %p, i32 %add
  store i32 %i, i32* %arrayidx, align 4
  ret void
}
