; RUN: opt --passes=verify %s -o - -S | FileCheck %s

;; Test that the type definitions are discovered when serialising to LLVM-IR,
;; even if they're only present inside a DbgRecord, and thus not normally
;; visible.

; CHECK: %union.anon = type { %struct.a }
; CHECK: %struct.a = type { i32 }
; CHECK: %union.anon2 = type { %struct.a2 }
; CHECK: %struct.a2 = type { i32 }

; ModuleID = 'bbi-98372.ll'
source_filename = "bbi-98372.ll"

%union.anon = type { %struct.a }
%struct.a = type { i32 }
%union.anon2 = type { %struct.a2 }
%struct.a2 = type { i32 }

@d = global [1 x { i16, i16 }] [{ i16, i16 } { i16 0, i16 undef }], align 1
@e = global [1 x { i16, i16 }] [{ i16, i16 } { i16 0, i16 undef }], align 1

define void @f() {
entry:
    #dbg_value(ptr getelementptr inbounds ([1 x %union.anon], ptr @d, i32 0, i32 3), !7, !DIExpression(), !14)
    #dbg_assign(ptr null, !7, !DIExpression(), !16, ptr getelementptr inbounds ([1 x %union.anon2], ptr @e, i32 0, i32 3), !17, !14)
  ret void, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "foo.c", directory: "/bar")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 1}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!"clang"}
!7 = !DILocalVariable(name: "f", scope: !8, file: !1, line: 8, type: !12)
!8 = distinct !DISubprogram(name: "e", scope: !1, file: !1, line: 8, type: !9, scopeLine: 8, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !{!7}
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 16)
!13 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
!14 = !DILocation(line: 0, scope: !8)
!15 = !DILocation(line: 8, column: 28, scope: !8)
!16 = distinct !DIAssignID()
!17 = !DIExpression()
