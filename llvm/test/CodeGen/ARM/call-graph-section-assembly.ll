;; Test if temporary labels are generated for each indirect callsite.
;; Test if the .llvm.callgraph section contains the MD5 hash of callees' type (type id)
;; is correctly paired with its corresponding temporary label generated for indirect
;; call sites annotated with !callee_type metadata.
;; Test if the .llvm.callgraph section contains unique direct callees.

; RUN: llc -mtriple=arm-unknown-linux --call-graph-section -o - < %s | FileCheck %s

declare !type !0 void @direct_foo()
declare !type !1 i32 @direct_bar(i8)
declare !type !2 ptr @direct_baz(ptr)

; CHECK: ball:
; CHECK-NEXT: [[LABEL_FUNC:\.Lfunc_begin[0-9]+]]:
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

; CHECK: .section .llvm.callgraph,"o",%progbits,.text
;; Version
; CHECK-NEXT: .byte   0
;; Flags
; CHECK-NEXT: .byte   7
;; Function Entry PC
; CHECK-NEXT: .long   [[LABEL_FUNC]]
;; Function type ID -- set to 0 as no type metadata attached to function.
; CHECK-NEXT: .long   0
; CHECK-NEXT: .long   0
;; Number of unique direct callees.
; CHECK-NEXT: .byte	  3
;; Direct callees.
; CHECK-NEXT: .long	direct_foo
; CHECK-NEXT: .long	direct_bar
; CHECK-NEXT: .long	direct_baz
;; Number of unique indirect target type IDs.
; CHECK-NEXT: .byte   3
;; Indirect type IDs.
; CHECK-NEXT: .long 838288420
; CHECK-NEXT: .long 1053552373
; CHECK-NEXT: .long 1505527380
; CHECK-NEXT: .long 814631809
; CHECK-NEXT: .long 342417018
; CHECK-NEXT: .long 2013108216
