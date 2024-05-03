; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck --check-prefix=SM20 %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_35 | FileCheck --check-prefix=SM35 %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_35 | %ptxas-verify %}


declare i32 @llvm.nvvm.rotate.b32(i32, i32)
declare i64 @llvm.nvvm.rotate.b64(i64, i32)
declare i64 @llvm.nvvm.rotate.right.b64(i64, i32)

; SM20: rotate32
; SM35: rotate32
define i32 @rotate32(i32 %a, i32 %b) {
; SM20: shl.b32
; SM20: sub.s32
; SM20: shr.b32
; SM20: add.u32
; SM35: shf.l.wrap.b32
  %val = tail call i32 @llvm.nvvm.rotate.b32(i32 %a, i32 %b)
  ret i32 %val
}

; SM20: rotate64
; SM35: rotate64
define i64 @rotate64(i64 %a, i32 %b) {
; SM20: shl.b64
; SM20: sub.u32
; SM20: shr.b64
; SM20: add.u64
; SM35: shf.l.wrap.b32
; SM35: shf.l.wrap.b32
  %val = tail call i64 @llvm.nvvm.rotate.b64(i64 %a, i32 %b)
  ret i64 %val
}

; SM20: rotateright64
; SM35: rotateright64
define i64 @rotateright64(i64 %a, i32 %b) {
; SM20: shr.b64
; SM20: sub.u32
; SM20: shl.b64
; SM20: add.u64
; SM35: shf.r.wrap.b32
; SM35: shf.r.wrap.b32
  %val = tail call i64 @llvm.nvvm.rotate.right.b64(i64 %a, i32 %b)
  ret i64 %val
}

; SM20: rotl0
; SM35: rotl0
define i32 @rotl0(i32 %x) {
; SM20: shl.b32
; SM20: shr.b32
; SM20: add.u32
; SM35: shf.l.wrap.b32
  %t0 = shl i32 %x, 8
  %t1 = lshr i32 %x, 24
  %t2 = or i32 %t0, %t1
  ret i32 %t2
}
