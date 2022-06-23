;; Test that we keep SHF_LINK_ORDER but reset sh_link to 0 if the associated
;; symbol is not defined.
; RUN: llc -mtriple=x86_64 -data-sections=1 < %s | FileCheck %s
; RUN: llc -filetype=obj -mtriple=x86_64 -data-sections=1 < %s | llvm-readelf -S - | FileCheck --check-prefix=SEC %s

;; FIXME The assembly output cannot be assembled because foo is not defined.
;; This is difficult to fix because we allow loops (see elf-associated.ll
;; .data.c and .data.d).
; CHECK: .section .data.a,"awo",@progbits,foo
; CHECK: .section .data.b,"awo",@progbits,foo

;; No 'L' (SHF_LINK_ORDER). sh_link=0.
; SEC: Name    {{.*}} Flg Lk Inf
; SEC: .data.a {{.*}} WAL  0   0
; SEC: .data.b {{.*}} WAL  0   0

;; The definition may be discarded by LTO.
declare void @foo()

@a = global i32 1, !associated !0
@b = global i32 1, !associated !0

!0 = !{ptr @foo}
