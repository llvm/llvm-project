; RUN: opt -S -passes=verify < %s | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

;; Check supported instructions are accepted without dropping 'vscale'.
;; Same order as the LangRef

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Unary Operations
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


define <vscale x 2 x double> @fneg(<vscale x 2 x double> %val) {
; CHECK-LABEL: @fneg
; CHECK: %r = fneg <vscale x 2 x double> %val
; CHECK-NEXT: ret <vscale x 2 x double> %r
  %r = fneg <vscale x 2 x double> %val
  ret <vscale x 2 x double> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Binary Operations
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define <vscale x 8 x i16> @add(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: @add
; CHECK: %r = add <vscale x 8 x i16> %a, %b
; CHECK-NEXT: ret <vscale x 8 x i16> %r
  %r = add <vscale x 8 x i16> %a, %b
  ret <vscale x 8 x i16> %r
}

define <vscale x 4 x float> @fadd(<vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: @fadd
; CHECK: %r = fadd <vscale x 4 x float> %a, %b
; CHECK-NEXT: ret <vscale x 4 x float> %r
  %r = fadd <vscale x 4 x float> %a, %b
  ret <vscale x 4 x float> %r
}

define <vscale x 4 x i32> @sub(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: @sub
; CHECK: %r = sub <vscale x 4 x i32> %a, %b
; CHECK-NEXT: ret <vscale x 4 x i32> %r
  %r = sub <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %r
}

define <vscale x 4 x float> @fsub(<vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: @fsub
; CHECK: %r = fsub <vscale x 4 x float> %a, %b
; CHECK-NEXT: ret <vscale x 4 x float> %r
  %r = fsub <vscale x 4 x float> %a, %b
  ret <vscale x 4 x float> %r
}

define <vscale x 4 x i32> @mul(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: @mul
; CHECK: %r = mul <vscale x 4 x i32> %a, %b
; CHECK-NEXT: ret <vscale x 4 x i32> %r
  %r = mul <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %r
}

define <vscale x 4 x float> @fmul(<vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: @fmul
; CHECK: %r = fmul <vscale x 4 x float> %a, %b
; CHECK-NEXT: ret <vscale x 4 x float> %r
  %r = fmul <vscale x 4 x float> %a, %b
  ret <vscale x 4 x float> %r
}

define <vscale x 4 x i32> @udiv(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: @udiv
; CHECK: %r = udiv <vscale x 4 x i32> %a, %b
; CHECK-NEXT: ret <vscale x 4 x i32> %r
  %r = udiv <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %r
}

define <vscale x 4 x i32> @sdiv(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: @sdiv
; CHECK: %r = sdiv <vscale x 4 x i32> %a, %b
; CHECK-NEXT: ret <vscale x 4 x i32> %r
  %r = sdiv <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %r
}

define <vscale x 4 x float> @fdiv(<vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: @fdiv
; CHECK: %r = fdiv <vscale x 4 x float> %a, %b
; CHECK-NEXT: ret <vscale x 4 x float> %r
  %r = fdiv <vscale x 4 x float> %a, %b
  ret <vscale x 4 x float> %r
}

define <vscale x 4 x i32> @urem(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: @urem
; CHECK: %r = urem <vscale x 4 x i32> %a, %b
; CHECK-NEXT: ret <vscale x 4 x i32> %r
  %r = urem <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %r
}

define <vscale x 4 x i32> @srem(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: @srem
; CHECK: %r = srem <vscale x 4 x i32> %a, %b
; CHECK-NEXT: ret <vscale x 4 x i32> %r
  %r = srem <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %r
}

define <vscale x 4 x float> @frem(<vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: @frem
; CHECK: %r = frem <vscale x 4 x float> %a, %b
; CHECK-NEXT: ret <vscale x 4 x float> %r
  %r = frem <vscale x 4 x float> %a, %b
  ret <vscale x 4 x float> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Bitwise Binary Operations
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define <vscale x 4 x i32> @shl(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: @shl
; CHECK: %r = shl <vscale x 4 x i32> %a, %b
; CHECK-NEXT: ret <vscale x 4 x i32> %r
  %r = shl <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %r
}

define <vscale x 4 x i32> @lshr(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: @lshr
; CHECK: %r = lshr <vscale x 4 x i32> %a, %b
; CHECK-NEXT: ret <vscale x 4 x i32> %r
  %r = lshr <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %r
}

define <vscale x 4 x i32> @ashr(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: @ashr
; CHECK: %r = ashr <vscale x 4 x i32> %a, %b
; CHECK-NEXT: ret <vscale x 4 x i32> %r
  %r = ashr <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %r
}

define <vscale x 4 x i32> @and(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: @and
; CHECK: %r = and <vscale x 4 x i32> %a, %b
; CHECK-NEXT: ret <vscale x 4 x i32> %r
  %r = and <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %r
}

define <vscale x 4 x i32> @or(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: @or
; CHECK: %r = or <vscale x 4 x i32> %a, %b
; CHECK-NEXT: ret <vscale x 4 x i32> %r
  %r = or <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %r
}

define <vscale x 4 x i32> @xor(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: @xor
; CHECK: %r = xor <vscale x 4 x i32> %a, %b
; CHECK-NEXT: ret <vscale x 4 x i32> %r
  %r = xor <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i32> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Vector Operations
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define i64 @extractelement(<vscale x 2 x i64> %val) {
; CHECK-LABEL: @extractelement
; CHECK: %r = extractelement <vscale x 2 x i64> %val, i32 0
; CHECK-NEXT: ret i64 %r
  %r = extractelement <vscale x 2 x i64> %val, i32 0
  ret i64 %r
}

define <vscale x 16 x i8> @insertelement(<vscale x 16 x i8> %vec, i8 %ins) {
; CHECK-LABEL: @insertelement
; CHECK: %r = insertelement <vscale x 16 x i8> %vec, i8 %ins, i32 0
; CHECK-NEXT: ret <vscale x 16 x i8> %r
  %r = insertelement <vscale x 16 x i8> %vec, i8 %ins, i32 0
  ret <vscale x 16 x i8> %r
}

define <vscale x 8 x half> @shufflevector(half %val) {
; CHECK-LABEL: @shufflevector
; CHECK: %insvec = insertelement <vscale x 8 x half> undef, half %val, i32 0
; CHECK-NEXT: %r = shufflevector <vscale x 8 x half> %insvec, <vscale x 8 x half> undef, <vscale x 8 x i32> zeroinitializer
; CHECK-NEXT: ret <vscale x 8 x half> %r
  %insvec = insertelement <vscale x 8 x half> undef, half %val, i32 0
  %r = shufflevector <vscale x 8 x half> %insvec, <vscale x 8 x half> undef, <vscale x 8 x i32> zeroinitializer
  ret <vscale x 8 x half> %r
}

define <vscale x 8 x half> @shufflevector2(half %val) {
; CHECK-LABEL: @shufflevector
; CHECK: %insvec = insertelement <vscale x 8 x half> poison, half %val, i32 0
; CHECK-NEXT: %r = shufflevector <vscale x 8 x half> %insvec, <vscale x 8 x half> undef, <vscale x 8 x i32> zeroinitializer
; CHECK-NEXT: ret <vscale x 8 x half> %r
  %insvec = insertelement <vscale x 8 x half> poison, half %val, i32 0
  %r = shufflevector <vscale x 8 x half> %insvec, <vscale x 8 x half> undef, <vscale x 8 x i32> zeroinitializer
  ret <vscale x 8 x half> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Memory Access and Addressing Operations
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define void @alloca() {
; CHECK-LABEL: @alloca
; CHECK: %vec = alloca <vscale x 4 x i32>
; CHECK-NEXT: ret void
  %vec = alloca <vscale x 4 x i32>
  ret void
}

define <vscale x 2 x double> @load(ptr %ptr) {
; CHECK-LABEL: @load
; CHECK: %r = load <vscale x 2 x double>, ptr %ptr
; CHECK-NEXT: ret <vscale x 2 x double> %r
  %r = load <vscale x 2 x double>, ptr %ptr
  ret <vscale x 2 x double> %r
}

define void @store(<vscale x 4 x i32> %data, ptr %ptr) {
; CHECK-LABEL: @store
; CHECK: store <vscale x 4 x i32> %data, ptr %ptr
; CHECK-NEXT: ret void
  store <vscale x 4 x i32> %data, ptr %ptr
  ret void
}

define ptr @getelementptr(ptr %base) {
; CHECK-LABEL: @getelementptr
; CHECK-NEXT: ret ptr %base
  ret ptr %base
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Conversion Operations
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define <vscale x 4 x i32> @truncto(<vscale x 4 x i64> %val) {
; CHECK-LABEL: @truncto
; CHECK: %r = trunc <vscale x 4 x i64> %val to <vscale x 4 x i32>
; CHECK-NEXT: ret <vscale x 4 x i32> %r
  %r = trunc <vscale x 4 x i64> %val to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %r
}

define <vscale x 2 x i64> @zextto(<vscale x 2 x i16> %val) {
; CHECK-LABEL: @zextto
; CHECK: %r = zext <vscale x 2 x i16> %val to <vscale x 2 x i64>
; CHECK-NEXT: ret <vscale x 2 x i64> %r
  %r = zext <vscale x 2 x i16> %val to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %r
}

define <vscale x 4 x i32> @sextto(<vscale x 4 x i8> %val) {
; CHECK-LABEL: @sextto
; CHECK: %r = sext <vscale x 4 x i8> %val to <vscale x 4 x i32>
; CHECK-NEXT: ret <vscale x 4 x i32> %r
  %r = sext <vscale x 4 x i8> %val to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %r
}

define <vscale x 4 x half> @fptruncto(<vscale x 4 x float> %val) {
; CHECK-LABEL: @fptruncto
; CHECK: %r = fptrunc <vscale x 4 x float> %val to <vscale x 4 x half>
; CHECK-NEXT: ret <vscale x 4 x half> %r
  %r = fptrunc <vscale x 4 x float> %val to <vscale x 4 x half>
  ret <vscale x 4 x half> %r
}

define <vscale x 2 x double> @fpextto(<vscale x 2 x half> %val) {
; CHECK-LABEL: @fpextto
; CHECK: %r = fpext <vscale x 2 x half> %val to <vscale x 2 x double>
; CHECK-NEXT: ret <vscale x 2 x double> %r
  %r = fpext <vscale x 2 x half> %val to <vscale x 2 x double>
  ret <vscale x 2 x double> %r
}

define <vscale x 4 x i32> @fptouito(<vscale x 4 x float> %val) {
; CHECK-LABEL: @fptoui
; CHECK: %r = fptoui <vscale x 4 x float> %val to <vscale x 4 x i32>
; CHECK-NEXT: ret <vscale x 4 x i32> %r
  %r = fptoui <vscale x 4 x float> %val to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %r
}

define <vscale x 4 x i32> @fptosito(<vscale x 4 x float> %val) {
; CHECK-LABEL: @fptosi
; CHECK: %r = fptosi <vscale x 4 x float> %val to <vscale x 4 x i32>
; CHECK-NEXT: ret <vscale x 4 x i32> %r
  %r = fptosi <vscale x 4 x float> %val to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %r
}

define <vscale x 4 x float> @uitofpto(<vscale x 4 x i32> %val) {
; CHECK-LABEL: @uitofp
; CHECK: %r = uitofp <vscale x 4 x i32> %val to <vscale x 4 x float>
; CHECK-NEXT: ret <vscale x 4 x float> %r
  %r = uitofp <vscale x 4 x i32> %val to <vscale x 4 x float>
  ret <vscale x 4 x float> %r
}

define <vscale x 4 x float> @sitofpto(<vscale x 4 x i32> %val) {
; CHECK-LABEL: @sitofp
; CHECK: %r = sitofp <vscale x 4 x i32> %val to <vscale x 4 x float>
; CHECK-NEXT: ret <vscale x 4 x float> %r
  %r = sitofp <vscale x 4 x i32> %val to <vscale x 4 x float>
  ret <vscale x 4 x float> %r
}

define <vscale x 2 x i64> @ptrtointto(<vscale x 2 x ptr> %val) {
; CHECK-LABEL: @ptrtointto
; CHECK: %r = ptrtoint <vscale x 2 x ptr> %val to <vscale x 2 x i64>
; CHECK-NEXT: ret <vscale x 2 x i64> %r
  %r = ptrtoint <vscale x 2 x ptr> %val to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %r
}

define <vscale x 2 x ptr> @inttoptrto(<vscale x 2 x i64> %val) {
; CHECK-LABEL: @inttoptrto
; CHECK: %r = inttoptr <vscale x 2 x i64> %val to <vscale x 2 x ptr>
; CHECK-NEXT: ret <vscale x 2 x ptr> %r
  %r = inttoptr <vscale x 2 x i64> %val to <vscale x 2 x ptr>
  ret <vscale x 2 x ptr> %r
}

define <vscale x 2 x i64> @bitcastto(<vscale x 2 x double> %a) {
; CHECK-LABEL: @bitcast
; CHECK: %r = bitcast <vscale x 2 x double> %a to <vscale x 2 x i64>
; CHECK-NEXT: ret <vscale x 2 x i64> %r
  %r = bitcast <vscale x 2 x double> %a to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Other Operations
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define <vscale x 4 x i1> @icmp(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: @icmp
; CHECK: %r = icmp eq <vscale x 4 x i32> %a, %b
; CHECK-NEXT: ret <vscale x 4 x i1> %r
  %r = icmp eq <vscale x 4 x i32> %a, %b
  ret <vscale x 4 x i1> %r
}

define <vscale x 4 x i1> @fcmp(<vscale x 4 x float> %a, <vscale x 4 x float> %b) {
; CHECK-LABEL: @fcmp
; CHECK: %r = fcmp une <vscale x 4 x float> %a, %b
; CHECK-NEXT: ret <vscale x 4 x i1> %r
  %r = fcmp une <vscale x 4 x float> %a, %b
  ret <vscale x 4 x i1> %r
}

define <vscale x 16 x i8> @phi(<vscale x 16 x i8> %a, i32 %val) {
; CHECK-LABEL: @phi
; CHECK: %r = phi <vscale x 16 x i8> [ %a, %entry ], [ %added, %iszero ]
; CHECK-NEXT: ret <vscale x 16 x i8> %r
entry:
  %cmp = icmp eq i32 %val, 0
  br i1 %cmp, label %iszero, label %end

iszero:
  %ins = insertelement <vscale x 16 x i8> undef, i8 1, i32 0
  %splatone = shufflevector <vscale x 16 x i8> %ins, <vscale x 16 x i8> undef, <vscale x 16 x i32> zeroinitializer
  %added = add <vscale x 16 x i8> %a, %splatone
  br label %end

end:
  %r = phi <vscale x 16 x i8> [ %a, %entry ], [ %added, %iszero ]
  ret <vscale x 16 x i8> %r
}

define <vscale x 8 x half> @select(<vscale x 8 x half> %a, <vscale x 8 x half> %b, <vscale x 8 x i1> %sval) {
; CHECK-LABEL: @select
; CHECK: %r = select <vscale x 8 x i1> %sval, <vscale x 8 x half> %a, <vscale x 8 x half> %b
; CHECK-NEXT: ret <vscale x 8 x half> %r
  %r = select <vscale x 8 x i1> %sval, <vscale x 8 x half> %a, <vscale x 8 x half> %b
  ret <vscale x 8 x half> %r
}

declare <vscale x 4 x i32> @callee(<vscale x 4 x i32>)
define <vscale x 4 x i32> @call(<vscale x 4 x i32> %val) {
; CHECK-LABEL: @call
; CHECK: %r = call <vscale x 4 x i32> @callee(<vscale x 4 x i32> %val)
; CHECK-NEXT: ret <vscale x 4 x i32> %r
  %r = call <vscale x 4 x i32> @callee(<vscale x 4 x i32> %val)
  ret <vscale x 4 x i32> %r
}
