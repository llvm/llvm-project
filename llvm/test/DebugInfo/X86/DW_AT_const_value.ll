; RUN: llc %s -o - | FileCheck %s

;; Check single location variables of various types with a constant value are
;; emitted with a DW_AT_const_value attribute.

;; DW_ATE_complex_float
;; Modified from C source input:
;; #include <complex.h>
;; void f() { double complex r1; }
; CHECK: .Ldebug_info_start0:
; CHECK:      .byte [[#]]                  # Abbrev {{.*}} DW_TAG_variable
; CHECK-NEXT: .byte 0                      # DW_AT_const_value
; CHECK-NEXT: .byte [[r1_str_idx:[0-9]+]]  # DW_AT_name

;; DW_ATE_float
;; from (at -Os):
;; void foo() {
;;   float a = 3.14;
;;   *(int *)&a = 0;
;; }
; CHECK:      .byte [[#]]                  # Abbrev {{.*}} DW_TAG_variable
; CHECK-NEXT: .byte 0                      # DW_AT_const_value
; CHECK-NEXT: .byte [[a_str_idx:[0-9]+]]   # DW_AT_name

;; DW_ATE_signed (i128)
; CHECK:      .byte [[#]]                  # Abbrev {{.*}} DW_TAG_variable
; CHECK-NEXT: .byte 16                     # DW_AT_const_value
; CHECK-NEXT: .byte 42
; CHECK-COUNT-15: .byte 0
; CHECK-NEXT: .byte [[MAX_str_idx:[0-9]+]] # DW_AT_name

;; DW_ATE_signed (i32)
; CHECK:      .byte [[#]]                  # Abbrev {{.*}} DW_TAG_variable
; CHECK-NEXT: .byte 42                     # DW_AT_const_value
; CHECK-NEXT: .byte [[i_str_idx:[0-9]+]]   # DW_AT_name

; CHECK: .Linfo_string[[r1_str_idx]]:
; CHECK-NEXT: .asciz "r1"
; CHECK: .Linfo_string[[a_str_idx]]:
; CHECK-NEXT: .asciz "a"
; CHECK: .Linfo_string[[MAX_str_idx]]:
; CHECK-NEXT: .asciz "MAX"
; CHECK: .Linfo_string[[i_str_idx]]:
; CHECK-NEXT: .asciz "i"

target triple = "x86_64-unknown-linux-gnu"

define dso_local void @f() local_unnamed_addr !dbg !10 {
entry:
  ;; DW_ATE_complex_float
  call void @llvm.dbg.value(metadata i8 0, metadata !14, metadata !DIExpression()), !dbg !17
  ;; DW_ATE_float
  call void @llvm.dbg.declare(metadata ptr undef, metadata !20, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 1078523331, metadata !20, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.value(metadata i32 0, metadata !20, metadata !DIExpression()), !dbg !17
  ;; DW_ATE_signed
  call void @llvm.dbg.value(metadata i128 42 , metadata !22, metadata !DIExpression()), !dbg !17
  ;; DW_ATE_signed
  call void @llvm.dbg.value(metadata i32 42, metadata !25, metadata !DIExpression()), !dbg !17
  ret void, !dbg !17
}

declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 17.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 17.0.0"}
!10 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !11, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !{!14}
!14 = !DILocalVariable(name: "r1", scope: !10, file: !1, line: 3, type: !15)
!15 = !DIBasicType(name: "complex", size: 128, encoding: DW_ATE_complex_float)
!17 = !DILocation(line: 0, scope: !10)
!20 = !DILocalVariable(name: "a", scope: !10, file: !1, line: 3, type: !21)
!21 = !DIBasicType(tag: DW_TAG_base_type, name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!22 = !DILocalVariable(name: "MAX", line: 29, scope: !10, file: !1, type: !23)
!23 = !DIDerivedType(tag: DW_TAG_typedef, name: "ti_int", line: 78, file: !1, scope: !1, baseType: !24)
!24 = !DIBasicType(tag: DW_TAG_base_type, size: 128, align: 128, encoding: DW_ATE_signed)
!25 = !DILocalVariable(name: "i", line: 2, scope: !10, file: !1, type: !26)
!26 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
