; RUN: opt -passes=argpromotion -S < %s | FileCheck %s

; ArgumentPromotion should describe a promoted pointer argument with
; DW_OP_LLVM_implicit_pointer only when the promoted value is the whole
; pointee and the expression does no computation on the address. Anything
; else must be poisoned rather than emitting wrong debug info.

%struct.t = type { i32, i32 }

; Whole pointee promoted: int *p -> int.
; CHECK-LABEL: define internal i32 @whole(i32 %p.0.val)
; CHECK:         #dbg_value(i32 %p.0.val, ![[VW:[0-9]+]], !DIExpression(DW_OP_LLVM_implicit_pointer),
define internal i32 @whole(ptr %p) !dbg !20 {
entry:
    #dbg_value(ptr %p, !30, !DIExpression(), !40)
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

; Only p->a promoted: offset 0, but smaller than struct t.
; CHECK-LABEL: define internal i32 @field_a(i32 %p.0.val)
; CHECK:         #dbg_value(ptr poison, ![[VA:[0-9]+]], !DIExpression(),
define internal i32 @field_a(ptr %p) !dbg !21 {
entry:
    #dbg_value(ptr %p, !31, !DIExpression(), !41)
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

; Only p->b promoted (offset 4): the promoted value is not *p.
; CHECK-LABEL: define internal i32 @field_b(i32 %p.4.val)
; CHECK:         #dbg_value(ptr poison, ![[VB:[0-9]+]], !DIExpression(),
define internal i32 @field_b(ptr %p) !dbg !22 {
entry:
    #dbg_value(ptr %p, !32, !DIExpression(), !42)
  %b = getelementptr inbounds i8, ptr %p, i64 4
  %v = load i32, ptr %b, align 4
  ret i32 %v
}

; The expression already computes on the address.
; CHECK-LABEL: define internal i32 @complex(i32 %p.0.val)
; CHECK:         #dbg_value(ptr poison, ![[VC:[0-9]+]], !DIExpression(DW_OP_plus_uconst, 4),
define internal i32 @complex(ptr %p) !dbg !23 {
entry:
    #dbg_value(ptr %p, !33, !DIExpression(DW_OP_plus_uconst, 4), !43)
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

define i32 @bar(i32 %x) !dbg !24 {
entry:
  %i = alloca i32, align 4
  %s = alloca %struct.t, align 4
  store i32 %x, ptr %i, align 4
  store i32 %x, ptr %s, align 4
  %s.b = getelementptr inbounds i8, ptr %s, i64 4
  store i32 %x, ptr %s.b, align 4
  %r0 = call i32 @whole(ptr %i), !dbg !44
  %r1 = call i32 @field_a(ptr %s), !dbg !44
  %r2 = call i32 @field_b(ptr %s), !dbg !44
  %r3 = call i32 @complex(ptr %i), !dbg !44
  %a0 = add i32 %r0, %r1
  %a1 = add i32 %r2, %r3
  %a2 = add i32 %a0, %a1
  ret i32 %a2
}

; CHECK-DAG: ![[VW]] = !DILocalVariable(name: "p"
; CHECK-DAG: ![[VA]] = !DILocalVariable(name: "p"
; CHECK-DAG: ![[VB]] = !DILocalVariable(name: "p"
; CHECK-DAG: ![[VC]] = !DILocalVariable(name: "p"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}

!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: !1, line: 1, size: 64, elements: !9)
!9 = !{!10, !11}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !8, file: !1, line: 1, baseType: !6, size: 32)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !8, file: !1, line: 1, baseType: !6, size: 32, offset: 32)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!13 = !DISubroutineType(types: !14)
!14 = !{!6, !7}
!15 = !DISubroutineType(types: !16)
!16 = !{!6, !12}
!17 = !DISubroutineType(types: !18)
!18 = !{!6, !6}

!20 = distinct !DISubprogram(name: "whole", scope: !1, file: !1, line: 2, type: !13, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!21 = distinct !DISubprogram(name: "field_a", scope: !1, file: !1, line: 3, type: !15, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!22 = distinct !DISubprogram(name: "field_b", scope: !1, file: !1, line: 4, type: !15, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!23 = distinct !DISubprogram(name: "complex", scope: !1, file: !1, line: 5, type: !13, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!24 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 6, type: !17, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)

!30 = !DILocalVariable(name: "p", arg: 1, scope: !20, file: !1, line: 2, type: !7)
!31 = !DILocalVariable(name: "p", arg: 1, scope: !21, file: !1, line: 3, type: !12)
!32 = !DILocalVariable(name: "p", arg: 1, scope: !22, file: !1, line: 4, type: !12)
!33 = !DILocalVariable(name: "p", arg: 1, scope: !23, file: !1, line: 5, type: !7)

!40 = !DILocation(line: 2, scope: !20)
!41 = !DILocation(line: 3, scope: !21)
!42 = !DILocation(line: 4, scope: !22)
!43 = !DILocation(line: 5, scope: !23)
!44 = !DILocation(line: 6, scope: !24)
