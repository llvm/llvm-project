; RUN: split-file %s %t

; RUN: opt -module-summary %t/av_ext_def.ll -o %t/av_ext_def.bc
; RUN: opt -module-summary %t/weak_def.ll -o %t/weak_def.bc
; RUN: llvm-lto2 run -o %t/prevailing_import -save-temps %t/av_ext_def.bc %t/weak_def.bc \
; RUN:   -r=%t/av_ext_def.bc,ret_av_ext_def,px -r=%t/av_ext_def.bc,def,x \
; RUN:   -r=%t/weak_def.bc,ret_weak_def,px -r=%t/weak_def.bc,def,px
; RUN: llvm-dis %t/prevailing_import.2.3.import.bc -o - | FileCheck --match-full-lines --check-prefix=WEAK_DEF %s
; RUN: llvm-nm -jU %t/prevailing_import.2 | FileCheck --match-full-lines --check-prefix=NM %s

;; def should remain weak after function importing in the weak_def module
; WEAK_DEF: @def = weak constant i32 0

;; It should also be defined in the corresponding object file
; NM: def

;--- av_ext_def.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
@def = available_externally constant i32 0
define ptr @ret_av_ext_def() {
  ret ptr @def
}

;--- weak_def.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
@def = weak constant i32 0
define ptr @ret_weak_def() {
  ret ptr @def
}
