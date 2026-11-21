; RUN: llc -O0 -mtriple=aarch64 -global-isel -global-isel-abort=1 -verify-machineinstrs < %s -o /dev/null
; RUN: llc -O0 -mtriple=aarch64 -global-isel -global-isel-abort=1 -stop-after=ir-translator < %s | FileCheck %s

; CHECK-LABEL: name: sdiv_v3i8
; CHECK-NOT: G_INTRINSIC intrinsic(@llvm.masked
; CHECK-COUNT-3: G_SDIV
; CHECK-COUNT-3: G_FREEZE
define void @sdiv_v3i8(ptr %p, <3 x i8> %a, <3 x i8> %b, <3 x i1> %m) {
  %r = call <3 x i8> @llvm.masked.sdiv.v3i8(<3 x i8> %a, <3 x i8> %b, <3 x i1> %m)
  %f = freeze <3 x i8> %r
  store <3 x i8> %f, ptr %p
  ret void
}

; CHECK-LABEL: name: urem_v3i8
; CHECK-NOT: G_INTRINSIC intrinsic(@llvm.masked
; CHECK-COUNT-3: G_{{UDIV|UREM}}
define void @urem_v3i8(ptr %p, <3 x i8> %a, <3 x i8> %b, <3 x i1> %m) {
  %r = call <3 x i8> @llvm.masked.urem.v3i8(<3 x i8> %a, <3 x i8> %b, <3 x i1> %m)
  %f = freeze <3 x i8> %r
  store <3 x i8> %f, ptr %p
  ret void
}

; CHECK-LABEL: name: srem_v2i64
; CHECK-NOT: G_INTRINSIC intrinsic(@llvm.masked
; CHECK-COUNT-2: G_{{SDIV|SREM}}
define void @srem_v2i64(ptr %p, <2 x i64> %a, <2 x i64> %b, <2 x i1> %m) {
  %r = call <2 x i64> @llvm.masked.srem.v2i64(<2 x i64> %a, <2 x i64> %b, <2 x i1> %m)
  %f = freeze <2 x i64> %r
  store <2 x i64> %f, ptr %p
  ret void
}

; CHECK-LABEL: name: freeze_add_v4i32
; CHECK:       [[ADD:%[0-9]+]]:_(<4 x i32>) = G_ADD
; CHECK-NEXT:  {{%[0-9]+}}:_(<4 x i32>) = G_FREEZE [[ADD]]
; CHECK-NOT:   G_FREEZE
define void @freeze_add_v4i32(ptr %p, <4 x i32> %a, <4 x i32> %b) {
  %s = add <4 x i32> %a, %b
  %f = freeze <4 x i32> %s
  store <4 x i32> %f, ptr %p
  ret void
}

; CHECK-LABEL: name: freeze_abs_v4i32
; CHECK:       [[ABS:%[0-9]+]]:_(<4 x i32>) = G_ABS
; CHECK-NEXT:  {{%[0-9]+}}:_(<4 x i32>) = G_FREEZE [[ABS]]
; CHECK-NOT:   G_FREEZE
define void @freeze_abs_v4i32(ptr %p, <4 x i32> %a) {
  %s = call <4 x i32> @llvm.abs.v4i32(<4 x i32> %a, i1 false)
  %f = freeze <4 x i32> %s
  store <4 x i32> %f, ptr %p
  ret void
}

; CHECK-LABEL: name: freeze_scalar_sdiv
; CHECK:       [[DIV:%[0-9]+]]:_(i32) = G_SDIV
; CHECK-NEXT:  {{%[0-9]+}}:_(i32) = G_FREEZE [[DIV]]
define i32 @freeze_scalar_sdiv(i32 %a, i32 %b, i1 %m) {
  %r = call i32 @llvm.masked.sdiv.i32(i32 %a, i32 %b, i1 %m)
  %f = freeze i32 %r
  ret i32 %f
}
