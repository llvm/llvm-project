; RUN: llc < %s -fast-isel -relocation-model=pic | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

@glob.hwasan = private constant i64 0

;; The constant here is 0x2F << 56. This effectively makes the alias a tagged version of the original global.
@glob = private alias i64, inttoptr (i64 add (i64 ptrtoint (ptr @glob.hwasan to i64), i64 3386706919782612992) to ptr)

; CHECK-LABEL: func
define void @func() #0 {
entry:
  ; CHECK:      adrp    [[REG:x[0-9]+]], :pg_hi21_nc:.Lglob
  ; CHECK-NEXT: movk    [[REG]], #:prel_g3:.Lglob+4294967296
  ; CHECK-NEXT: add     x0, [[REG]], :lo12:.Lglob
  call void @extern_func(ptr @glob)
  ret void
}

declare void @extern_func(ptr)

attributes #0 = { "target-features"="+tagged-globals" }
