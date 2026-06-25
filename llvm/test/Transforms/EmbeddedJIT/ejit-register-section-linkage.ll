; RUN: opt -passes=ejit-register-bitcode -S %s | FileCheck %s --check-prefix=BC
; RUN: opt -passes=ejit-register-period -S %s | FileCheck %s --check-prefix=PD
;
; Multi-TU link safety: PASS1/PASS2 must emit their per-translation-unit
; registry entries as PRIVATE (local) globals placed in dedicated, C-identifier
; named sections (".ejit_bitcode" / ".ejit_period") and kept alive via llvm.used.
; The linker concatenates these sections across all TUs and synthesizes the
; __start_/__stop_ bounds the runtime walks.
;
; Previously each TU emitted a single EXTERNAL global with the fixed name
; __ejit_registry_bitcode / __ejit_registry_period, which caused
;     ld.lld: error: duplicate symbol
; as soon as more than one .c file defined ejit_entry functions. Neither fixed
; external name may reappear.

@cells = global [10 x i32] zeroinitializer, !ejit.metadata !10
@thresh = global i32 5, !ejit.metadata !11

define void @helper() {
  ret void
}

define void @my_entry() !ejit.metadata !0 {
  call void @helper()
  ret void
}

!0 = distinct !{!{!"ejit_entry"}}
!10 = distinct !{!{!"ejit_period_arr", !"cell", i32 10}}
!11 = distinct !{!{!"ejit_period", !"static"}}

; --- PASS1: bitcode registry ---
; The table is private (local symbol -> no cross-TU collision), lives in the
; ejit_bitcode section, and is anchored by llvm.used. No fixed external symbol.
; BC: @{{.*}} = private constant {{.*}}, section ".ejit_bitcode"
; BC: @llvm.used = appending global {{.*}} section "llvm.metadata"
; BC-NOT: @__ejit_registry_bitcode

; --- PASS2: period registry ---
; PD: @{{.*}} = private constant {{.*}}, section ".ejit_period"
; PD: @llvm.used = appending global {{.*}} section "llvm.metadata"
; PD-NOT: @__ejit_registry_period
