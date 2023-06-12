; RUN: llc %s -o - | FileCheck %s

;; Check a single location variable of type DW_ATE_complex_float with a
;; constant value is emitted with a DW_AT_const_value attribute.

;; Modified from C source input:
;; #include <complex.h>
;; void f() { double complex r1; }

; CHECK: .Ldebug_info_start0:
; CHECK:      .byte [[#]]              # Abbrev {{.*}} DW_TAG_variable
; CHECK-NEXT: .byte 0                  # DW_AT_const_value
; CHECK-NEXT: .byte [[str_idx:[0-9]+]] # DW_AT_name

; CHECK: .Linfo_string[[str_idx]]:
; CHECK-NEXT: .asciz "r1"

target triple = "x86_64-unknown-linux-gnu"

define dso_local void @f() local_unnamed_addr !dbg !10 {
entry:
  call void @llvm.dbg.value(metadata i8 0, metadata !14, metadata !DIExpression()), !dbg !17
  ret void, !dbg !19
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

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
!19 = !DILocation(line: 4, column: 1, scope: !10)
