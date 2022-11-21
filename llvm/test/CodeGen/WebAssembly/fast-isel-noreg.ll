; RUN: llc < %s -asm-verbose=false -mcpu=mvp -disable-wasm-fallthrough-return-opt -wasm-keep-registers -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -asm-verbose=false -mcpu=mvp -wasm-keep-registers -fast-isel -verify-machineinstrs | FileCheck %s

; Test that FastISel does not generate instructions with NoReg

target triple = "wasm32-unknown-unknown"

; CHECK: i32.const $push0=, 0
define hidden i32 @a() #0 {
entry:
  ret i32 zext (i1 icmp eq (void (...)* inttoptr (i32 10 to void (...)*), void (...)* null) to i32)
}

; CHECK: i32.const $push0=, 1
; CHECK: br_if 0, $pop0
define hidden i32 @b() #0 {
entry:
  br i1 icmp eq (void (...)* inttoptr (i32 10 to void (...)*), void (...)* null), label %a, label %b
a:
  unreachable
b:
  ret i32 0
}

; CHECK: i32.const $push1=, 0
; CHECK: i32.const $push2=, 0
; CHECK: i32.store 0($pop1), $pop2
define hidden i32 @c() #0 {
entry:
  store i32 zext (i1 icmp eq (void (...)* inttoptr (i32 10 to void (...)*), void (...)* null) to i32), i32* inttoptr (i32 0 to i32 *)
  ret i32 0
}

; CHECK: i32.const {{.*}}, addr
; CHECK: i32.const {{.*}}, 24
; CHECK: i32.shl
; CHECK: i32.const {{.*}}, 24
; CHECK: i32.shr_s
; CHECK: i32.const {{.*}}, 64
; CHECK: i32.lt_s
; CHECK: i32.const {{.*}}, 1
; CHECK: i32.and
; CHECK: i32.eqz
; CHECK: br_if 0, $pop{{[0-9]+}}
define hidden i32 @d() #0 {
entry:
  %t = icmp slt i8 ptrtoint (void ()* @addr to i8), 64
  br i1 %t, label %a, label %b
a:
  unreachable
b:
  ret i32 0
}

; CHECK: i32.const {{.*}}, addr
; CHECK: i32.const {{.*}}, 255
; CHECK: i32.and
; CHECK: i32.const {{.*}}, 64
; CHECK: i32.lt_u
; CHECK: i32.const {{.*}}, 1
; CHECK: i32.and
; CHECK: i32.eqz
; CHECK: br_if 0, $pop{{[0-9]+}}
define hidden i32 @e() #0 {
entry:
  %t = icmp ult i8 ptrtoint (void ()* @addr to i8), 64
  br i1 %t, label %a, label %b
a:
  unreachable
b:
  ret i32 0
}

; CHECK: i32.const {{.*}}, addr
; CHECK: i32.const {{.*}}, 24
; CHECK: i32.shl
; CHECK: i32.const {{.*}}, 24
; CHECK: i32.shr_s
define hidden i32 @f() #0 {
entry:
  %t = sext i8 ptrtoint (void ()* @addr to i8) to i32
  ret i32 %t
}

; CHECK: i32.const {{.*}}, addr
; CHECK: i32.const {{.*}}, 255
; CHECK: i32.and
define hidden i32 @g() #0 {
entry:
  %t = zext i8 ptrtoint (void ()* @addr to i8) to i32
  ret i32 %t
}

declare void @addr()

attributes #0 = { noinline optnone }
