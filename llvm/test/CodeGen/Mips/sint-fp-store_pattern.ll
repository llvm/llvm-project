; RUN: llc -mtriple=mipsel < %s | FileCheck %s -check-prefix=32
; RUN: llc -mtriple=mips64el -mcpu=mips4 < %s | FileCheck %s -check-prefix=64
; RUN: llc -mtriple=mips64el -mcpu=mips64 < %s | FileCheck %s -check-prefix=64

@gint_ = external global i32
@gLL_ = external global i64

; 32-LABEL: store_int_float_:
; 32: trunc.w.s $f[[R0:[0-9]+]], $f{{[0-9]+}}
; 32: swc1 $f[[R0]],

define void @store_int_float_(float %a) {
entry:
  %conv = fptosi float %a to i32
  store i32 %conv, ptr @gint_, align 4
  ret void
}

; 32-LABEL: store_int_double_:
; 32: trunc.w.d $f[[R0:[0-9]+]], $f{{[0-9]+}}
; 32: swc1 $f[[R0]],
; 64-LABEL: store_int_double_:
; 64: trunc.w.d $f[[R0:[0-9]+]], $f{{[0-9]+}}
; 64: swc1 $f[[R0]],

define void @store_int_double_(double %a) {
entry:
  %conv = fptosi double %a to i32
  store i32 %conv, ptr @gint_, align 4
  ret void
}

; 64-LABEL: store_LL_float_:
; 64: trunc.l.s $f[[R0:[0-9]+]], $f{{[0-9]+}}
; 64: sdc1 $f[[R0]],

define void @store_LL_float_(float %a) {
entry:
  %conv = fptosi float %a to i64
  store i64 %conv, ptr @gLL_, align 8
  ret void
}

; 64-LABEL: store_LL_double_:
; 64: trunc.l.d $f[[R0:[0-9]+]], $f{{[0-9]+}}
; 64: sdc1 $f[[R0]],

define void @store_LL_double_(double %a) {
entry:
  %conv = fptosi double %a to i64
  store i64 %conv, ptr @gLL_, align 8
  ret void
}
