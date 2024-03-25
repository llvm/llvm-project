; RUN: llc -mtriple=aarch64 < %s | FileCheck %s
; RUN: llc -mtriple=arm64-apple-darwin < %s | FileCheck %s --check-prefix=MACHO
; RUN: llc -filetype=obj -mtriple=aarch64 %s -o %t
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s --check-prefix=DBG

; MACHO:         bl      ___xray_CustomEvent
; MACHO:         bl      ___xray_CustomEvent
; MACHO:         bl      ___xray_TypedEvent
; MACHO:         bl      ___xray_TypedEvent

; CHECK-LABEL: customevent:
; CHECK:       Lxray_sled_1:
; CHECK-NEXT:    b       #24                             // Begin XRay custom event
; CHECK-NEXT:    stp     x0, x1, [sp, #-16]!
; CHECK-NEXT:    mov     x0, x0
; CHECK-NEXT:    mov     x1, x1
; CHECK-NEXT:    bl      __xray_CustomEvent
; CHECK-NEXT:    ldp     x0, x1, [sp], #16               // End XRay custom event
; CHECK-NEXT:  Ltmp[[#]]:
; CHECK-NEXT:    .loc    0 3 3                           // a.c:3:3
; CHECK-NEXT:  Lxray_sled_2:
; CHECK-NEXT:    b       #24                             // Begin XRay custom event
; CHECK-NEXT:    stp     x0, x1, [sp, #-16]!
; CHECK-NEXT:    mov     x0, x2
; CHECK-NEXT:    mov     x1, x3
; CHECK-NEXT:    bl      __xray_CustomEvent
; CHECK-NEXT:    ldp     x0, x1, [sp], #16               // End XRay custom event
; CHECK-NEXT:  .Ltmp[[#]]:
; CHECK-NEXT:    .loc    0 4 1                           // a.c:4:1
define void @customevent(ptr nocapture noundef readonly %e1, i64 noundef %s1, ptr nocapture noundef readonly %e2, i64 noundef %s2) "function-instrument"="xray-always" !dbg !11 {
entry:
  tail call void @llvm.xray.customevent(ptr %e1, i64 %s1), !dbg !22
  tail call void @llvm.xray.customevent(ptr %e2, i64 %s2), !dbg !23
  ret void, !dbg !24
}

; CHECK:         .xword  .Lxray_sled_1-.Ltmp[[#]]
; CHECK-NEXT:    .xword  .Lfunc_begin0-(.Ltmp[[#]]+8)
; CHECK-NEXT:    .byte   0x04
; CHECK-NEXT:    .byte   0x01
; CHECK-NEXT:    .byte   0x02
; CHECK-NEXT:    .zero   13
; CHECK:         .xword  .Lxray_sled_2-.Ltmp[[#]]
; CHECK-NEXT:    .xword  .Lfunc_begin0-(.Ltmp[[#]]+8)
; CHECK-NEXT:    .byte   0x04
; CHECK-NEXT:    .byte   0x01
; CHECK-NEXT:    .byte   0x02
; CHECK-NEXT:    .zero   13

; CHECK-LABEL: typedevent:
; CHECK:       .Lxray_sled_5:
; CHECK-NEXT:    b       #36                             // Begin XRay typed event
; CHECK-NEXT:    stp     x0, x1, [sp, #-32]!
; CHECK-NEXT:    str     x2, [sp, #16]
; CHECK-NEXT:    mov     x0, x0
; CHECK-NEXT:    mov     x1, x1
; CHECK-NEXT:    mov     x2, x2
; CHECK-NEXT:    bl      __xray_TypedEvent
; CHECK-NEXT:    ldr     x2, [sp, #16]
; CHECK-NEXT:    ldp     x0, x1, [sp], #32               // End XRay typed event
; CHECK-NEXT:  .Ltmp[[#]]:
; CHECK-NEXT:    .loc    0 8 3                           // a.c:8:3
; CHECK-NEXT:  .Lxray_sled_6:
; CHECK-NEXT:    b       #36                             // Begin XRay typed event
; CHECK-NEXT:    stp     x0, x1, [sp, #-32]!
; CHECK-NEXT:    str     x2, [sp, #16]
; CHECK-NEXT:    mov     x0, x2
; CHECK-NEXT:    mov     x1, x1
; CHECK-NEXT:    mov     x2, x0
; CHECK-NEXT:    bl      __xray_TypedEvent
; CHECK-NEXT:    ldr     x2, [sp, #16]
; CHECK-NEXT:    ldp     x0, x1, [sp], #32               // End XRay typed event
; CHECK-NEXT:  .Ltmp[[#]]:
; CHECK-NEXT:    .loc    0 9 1                           // a.c:9:1
define void @typedevent(i64 noundef %type, ptr nocapture noundef readonly %event, i64 noundef %size) "function-instrument"="xray-always" !dbg !25 {
entry:
  tail call void @llvm.xray.typedevent(i64 %type, ptr %event, i64 %size), !dbg !33
  tail call void @llvm.xray.typedevent(i64 %size, ptr %event, i64 %type), !dbg !34
  ret void, !dbg !35
}

; CHECK:         .xword  .Lxray_sled_5-.Ltmp[[#]]
; CHECK-NEXT:    .xword  .Lfunc_begin1-(.Ltmp[[#]]+8)
; CHECK-NEXT:    .byte   0x05
; CHECK-NEXT:    .byte   0x01
; CHECK-NEXT:    .byte   0x02
; CHECK-NEXT:    .zero   13
; CHECK:         .xword  .Lxray_sled_6-.Ltmp[[#]]
; CHECK-NEXT:    .xword  .Lfunc_begin1-(.Ltmp[[#]]+8)
; CHECK-NEXT:    .byte   0x05
; CHECK-NEXT:    .byte   0x01
; CHECK-NEXT:    .byte   0x02
; CHECK-NEXT:    .zero   13

;; Construct call site entries for PATCHABLE_EVENT_CALL.
; DBG:      DW_TAG_subprogram
; DBG:      DW_AT_name
; DBG-SAME:            ("customevent")
; DBG:        DW_TAG_call_site
; DBG-NEXT:     DW_AT_call_target (DW_OP_reg0 {{.*}})
; DBG-NEXT:     DW_AT_call_return_pc
; DBG-EMPTY:
; DBG:        DW_TAG_call_site
; DBG-NEXT:     DW_AT_call_target (DW_OP_reg2 {{.*}})
; DBG-NEXT:     DW_AT_call_return_pc

declare void @llvm.xray.customevent(ptr, i64)
declare void @llvm.xray.typedevent(i64, ptr, i64)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 17.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "a.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 17.0.0"}
!11 = distinct !DISubprogram(name: "customevent", scope: !1, file: !1, line: 1, type: !12, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !14, !15, !14, !15}
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!15 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!21 = !DILocation(line: 0, scope: !11)
!22 = !DILocation(line: 2, column: 3, scope: !11)
!23 = !DILocation(line: 3, column: 3, scope: !11)
!24 = !DILocation(line: 4, column: 1, scope: !11)
!25 = distinct !DISubprogram(name: "typedevent", scope: !1, file: !1, line: 6, type: !26, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!26 = !DISubroutineType(types: !27)
!27 = !{null, !15, !14, !15}
!32 = !DILocation(line: 0, scope: !25)
!33 = !DILocation(line: 7, column: 3, scope: !25)
!34 = !DILocation(line: 8, column: 3, scope: !25)
!35 = !DILocation(line: 9, column: 1, scope: !25)
