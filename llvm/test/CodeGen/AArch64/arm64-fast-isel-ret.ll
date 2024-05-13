; RUN: llc -O0 -fast-isel -fast-isel-abort=1 -verify-machineinstrs -mtriple=arm64-apple-darwin < %s | FileCheck %s

;; Test returns.
define void @t0() nounwind ssp {
entry:
; CHECK: t0
; CHECK: ret
  ret void
}

define i32 @t1(i32 %a) nounwind ssp {
entry:
; CHECK: t1
; CHECK: str w0, [sp, #12]
; CHECK-NEXT: ldr w0, [sp, #12]
; CHECK: ret
  %a.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  %tmp = load i32, ptr %a.addr, align 4
  ret i32 %tmp
}

define i64 @t2(i64 %a) nounwind ssp {
entry:
; CHECK: t2
; CHECK: str x0, [sp, #8]
; CHECK-NEXT: ldr x0, [sp, #8]
; CHECK: ret
  %a.addr = alloca i64, align 8
  store i64 %a, ptr %a.addr, align 8
  %tmp = load i64, ptr %a.addr, align 8
  ret i64 %tmp
}

define signext i16 @ret_i16(i16 signext %a) nounwind {
entry:
; CHECK: @ret_i16
; CHECK: sxth	w0, {{w[0-9]+}}
  %a.addr = alloca i16, align 1
  store i16 %a, ptr %a.addr, align 1
  %0 = load i16, ptr %a.addr, align 1
  ret i16 %0
}

define signext i8 @ret_i8(i8 signext %a) nounwind {
entry:
; CHECK: @ret_i8
; CHECK: sxtb	w0, {{w[0-9]+}}
  %a.addr = alloca i8, align 1
  store i8 %a, ptr %a.addr, align 1
  %0 = load i8, ptr %a.addr, align 1
  ret i8 %0
}

define signext i1 @ret_i1(i1 signext %a) nounwind {
entry:
; CHECK: @ret_i1
; CHECK: and [[REG:w[0-9]+]], {{w[0-9]+}}, #0x1
; CHECK: sbfx w0, [[REG]], #0, #1
  %a.addr = alloca i1, align 1
  store i1 %a, ptr %a.addr, align 1
  %0 = load i1, ptr %a.addr, align 1
  ret i1 %0
}
