; REQUIRES: object-emission
; RUN: %llc_dwarf %s -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s

;; https://github.com/llvm/llvm-project/issues/140362
;; Don't assert when emitting a complex integer type in DWARF.

;; C source:
;; int g;
;;
;; void foo(_Complex short c) { __builtin_memmove(&g, (char *)&c, 2); }
;;
;; void bar() { foo(0); }

; CHECK: DW_AT_type ([[complex:0x[0-9a-f]+]] "complex")

; CHECK: [[complex]]: DW_TAG_base_type
; CHECK-NEXT: DW_AT_name        ("complex")
; CHECK-NEXT: DW_AT_encoding    (0x80)
; CHECK-NEXT: DW_AT_byte_size   (0x04)

@g = dso_local local_unnamed_addr global i32 0, align 4, !dbg !0

define dso_local void @bar() local_unnamed_addr !dbg !18 {
entry:
    #dbg_value(i32 0, !21, !DIExpression(), !27)
  store i16 0, ptr @g, align 4, !dbg !29
  ret void, !dbg !30
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !8, line: 1, type: !9, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 22.0.0git", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4, globals: !7, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "/app/example.cpp", directory: "/app")
!4 = !{!5}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!6 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!7 = !{!0}
!8 = !DIFile(filename: "example.cpp", directory: "/app")
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{i32 7, !"Dwarf Version", i32 5}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{!"clang version 22.0.0git"}
!18 = distinct !DISubprogram(name: "bar", linkageName: "bar()", scope: !8, file: !8, line: 5, type: !19, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, keyInstructions: true)
!19 = !DISubroutineType(types: !20)
!20 = !{null}
!21 = !DILocalVariable(name: "c", arg: 1, scope: !22, file: !8, line: 3, type: !25)
!22 = distinct !DISubprogram(name: "foo", linkageName: "_ZL3fooCs", scope: !8, file: !8, line: 3, type: !23, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !26, keyInstructions: true)
!23 = !DISubroutineType(types: !24)
!24 = !{null, !25}
!25 = !DIBasicType(name: "complex", size: 32, encoding: 128)
!26 = !{!21}
!27 = !DILocation(line: 0, scope: !22, inlinedAt: !28)
!28 = distinct !DILocation(line: 5, column: 14, scope: !18)
!29 = !DILocation(line: 3, column: 37, scope: !22, inlinedAt: !28, atomGroup: 1, atomRank: 1)
!30 = !DILocation(line: 5, column: 22, scope: !18, atomGroup: 1, atomRank: 1)
