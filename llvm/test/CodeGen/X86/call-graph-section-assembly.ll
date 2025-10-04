;; Test if temporary labels are generated for each indirect callsite.
;; Test if the .callgraph section contains the MD5 hash of callees' type (type id)
;; is correctly paired with its corresponding temporary label generated for indirect
;; call sites annotated with !callee_type metadata.
;; Test if the .callgraph section contains unique direct callees.

; REQUIRES: x86-registered-target
; REQUIRES: arm-registered-target

; RUN: llc -mtriple=x86_64-unknown-linux --call-graph-section -o - < %s | FileCheck --check-prefix=X64 %s
; RUN: llc -mtriple=arm-unknown-linux --call-graph-section -o - < %s | FileCheck --check-prefix=ARM32 %s

declare !type !0 void @direct_foo()
declare !type !1 i32 @direct_bar(i8)
declare !type !2 ptr @direct_baz(ptr)

; X64: ball:
; X64-NEXT: [[LABEL_FUNC:\.Lfunc_begin[0-9]+]]:
; ARM32: ball:
; ARM32-NEXT: [[LABEL_FUNC:\.Lfunc_begin[0-9]+]]:
define ptr @ball() {
entry:
  call void @direct_foo()
  %fp_foo_val = load ptr, ptr null, align 8   
  call void (...) %fp_foo_val(), !callee_type !0   
  call void @direct_foo()
  %fp_bar_val = load ptr, ptr null, align 8  
  %call_fp_bar = call i32 %fp_bar_val(i8 0), !callee_type !2  
  %call_fp_bar_direct = call i32 @direct_bar(i8 1)
  %fp_baz_val = load ptr, ptr null, align 8
  %call_fp_baz = call ptr %fp_baz_val(ptr null), !callee_type !4
  call void @direct_foo()
  %call_fp_baz_direct = call ptr @direct_baz(ptr null)
  call void @direct_foo()
  ret ptr %call_fp_baz
}

!0 = !{!1}
!1 = !{i64 0, !"_ZTSFvE.generalized"}
!2 = !{!3}
!3 = !{i64 0, !"_ZTSFicE.generalized"}
!4 = !{!5}
!5 = !{i64 0, !"_ZTSFPvS_E.generalized"}

; X64: .section .callgraph,"o",@progbits,.text
;; Version
; X64-NEXT: .byte   0
;; Flags
; X64-NEXT: .byte   7
;; Function Entry PC
; X64-NEXT: .quad   [[LABEL_FUNC]]
;; Function type ID -- set to 0 as no type metadata attached to function.
; X64-NEXT: .quad   0
;; Number of unique direct callees.
; X64-NEXT: .byte	  3
;; Direct callees.
; X64-NEXT: .quad	direct_foo
; X64-NEXT: .quad	direct_bar
; X64-NEXT: .quad	direct_baz
;; Number of unique indirect target type IDs.
; X64-NEXT: .byte   3
;; Indirect type IDs.
; X64-NEXT: .quad   4524972987496481828
; X64-NEXT: .quad   3498816979441845844
; X64-NEXT: .quad   8646233951371320954

; ARM32: .section .callgraph,"o",%progbits,.text
;; Version
; ARM32-NEXT: .byte   0
;; Flags
; ARM32-NEXT: .byte   7
;; Function Entry PC
; ARM32-NEXT: .long   [[LABEL_FUNC]]
;; Function type ID -- set to 0 as no type metadata attached to function.
; ARM32-NEXT: .long   0
; ARM32-NEXT: .long   0
;; Number of unique direct callees.
; ARM32-NEXT: .byte	  3
;; Direct callees.
; ARM32-NEXT: .long	direct_foo
; ARM32-NEXT: .long	direct_bar
; ARM32-NEXT: .long	direct_baz
;; Number of unique indirect target type IDs.
; ARM32-NEXT: .byte   3
;; Indirect type IDs.
; ARM32-NEXT: .long 838288420
; ARM32-NEXT: .long 1053552373
; ARM32-NEXT: .long 1505527380
; ARM32-NEXT: .long 814631809
; ARM32-NEXT: .long 342417018
; ARM32-NEXT: .long 2013108216
