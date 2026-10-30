; REQUIRES: object-emission
; RUN: llc -mtriple=x86_64-linux-gnu -filetype=obj -o %t %s
; RUN: llvm-dwarfdump --name=p %t | FileCheck %s

; DW_OP_LLVM_implicit_pointer is only lowered for single-location variables.
; When the variable needs a location list, the range carrying the implicit
; pointer must not be emitted as a (wrong) register/memory location, while the
; other ranges must be kept.
;
; Reduced from the following, after ArgumentPromotion (-O3):
;   struct S { int x; };
;   void use(int);
;   static int foo(struct S *p) {
;     int r = p->x + 5; use(r); p = 0; use(r); return r;
;   }

; CHECK:      DW_TAG_formal_parameter
; CHECK-NEXT:   DW_AT_location (indexed ({{.*}}) loclist = {{.*}}:
; CHECK-NEXT:     [0x{{[0-9a-f]+}}, 0x{{[0-9a-f]+}}): DW_OP_lit0, DW_OP_stack_value)
; CHECK-NEXT:   DW_AT_name ("p")

define dso_local i32 @foo(i32 %p.0.val) !dbg !10 {
entry:
    #dbg_value(i32 %p.0.val, !15, !DIExpression(DW_OP_LLVM_implicit_pointer), !17)
  %r = add nsw i32 %p.0.val, 5, !dbg !18
  tail call void @use(i32 %r), !dbg !18
    #dbg_value(ptr null, !15, !DIExpression(), !17)
  tail call void @use(i32 %r), !dbg !19
  ret i32 %r, !dbg !19
}

declare void @use(i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, line: 1, size: 32, elements: !6)
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !5, file: !1, line: 1, baseType: !4, size: 32)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!9 = !{!4, !8}
!10 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !11, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!11 = !DISubroutineType(types: !9)
!12 = !{!15}
!15 = !DILocalVariable(name: "p", arg: 1, scope: !10, file: !1, line: 3, type: !8)
!17 = !DILocation(line: 0, scope: !10)
!18 = !DILocation(line: 5, column: 3, scope: !10)
!19 = !DILocation(line: 7, column: 3, scope: !10)
