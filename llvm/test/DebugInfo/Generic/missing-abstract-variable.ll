; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; The formal parameter 'b' for Function 'x' when inlined within 'a' is lost on
; mips and powerpc64 (and on x86_64 at at least -O2). Presumably this is a
; SelectionDAG issue (do mips/powerpc64 use FastISel?).
; XFAIL: mips, powerpc64, s390x, sparc

; Build from the following source with clang -O2.

; The important details are that 'x's abstract definition is first built during
; the definition of 'b', where the parameter to 'x' is constant and so 'x's 's'
; variable is optimized away. No abstract definition DIE for 's' is constructed.
; Then, during 'a' emission, the abstract DbgVariable for 's' is created, but
; the abstract DIE isn't (since the abstract definition for 'b' is already
; built). This results in 's' inlined in 'a' being emitted with its name, line,
; file there, rather than referencing an abstract definition.

; extern int t;
;
; void f(int);
;
; inline void x(bool b) {
;   if (b) {
;     int s = t;
;     f(s);
;   }
;   f(0);
; }
;
; void b() {
;   x(false);
; }
;
; void a(bool u) {
;   x(u);
; }

; CHECK: [[X_DECL:.*]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "x"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "b"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:       DW_TAG_lexical_block
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:         DW_AT_name {{.*}} "s"

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "b"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_inlined_subroutine
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin {{.*}} {[[X_DECL]]}
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_abstract_origin {{.*}} "b"
; Notice 'x's local variable 's' is missing. Not necessarily a bug here,
; since it's been optimized entirely away and it should be described in
; abstract subprogram.
; CHECK-NOT: DW_TAG
; CHECK: NULL
; CHECK-NOT: DW_TAG
; CHECK: NULL

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "a"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_inlined_subroutine
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin {{.*}} {[[X_DECL]]}
; CHECK-NOT: {{DW_TAG|NULL}}
; FIXME: This formal parameter goes missing at least at -O2 (& on
; mips/powerpc), maybe before that. Perhaps SelectionDAG is to blame (and
; fastisel succeeds).
; CHECK:     DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_abstract_origin {{.*}} "b"

; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_TAG_lexical_block
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:       DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:         DW_AT_abstract_origin {{.*}} "s"

@t = external global i32

; Function Attrs: uwtable
define void @_Z1bv() #0 !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata i1 false, i64 0, metadata !25, metadata !DIExpression()), !dbg !27
  tail call void @_Z1fi(i32 0), !dbg !28
  ret void, !dbg !29
}

; Function Attrs: uwtable
define void @_Z1ab(i1 zeroext %u) #0 !dbg !8 {
entry:
  tail call void @llvm.dbg.value(metadata i1 %u, i64 0, metadata !13, metadata !DIExpression()), !dbg !30
  tail call void @llvm.dbg.value(metadata i1 %u, i64 0, metadata !31, metadata !DIExpression()), !dbg !33
  br i1 %u, label %if.then.i, label %_Z1xb.exit, !dbg !34

if.then.i:                                        ; preds = %entry
  %0 = load i32, i32* @t, align 4, !dbg !35, !tbaa !36
  tail call void @llvm.dbg.value(metadata i32 %0, i64 0, metadata !40, metadata !DIExpression()), !dbg !35
  tail call void @_Z1fi(i32 %0), !dbg !41
  br label %_Z1xb.exit, !dbg !42

_Z1xb.exit:                                       ; preds = %entry, %if.then.i
  tail call void @_Z1fi(i32 0), !dbg !43
  ret void, !dbg !44
}

declare void @_Z1fi(i32) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}
!llvm.ident = !{!23}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: true, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "missing-abstract-variables.cc", directory: "/tmp/dbginfo")
!2 = !{}
!3 = !{!4, !8, !14}
!4 = distinct !DISubprogram(name: "b", linkageName: "_Z1bv", line: 13, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 13, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "missing-abstract-variables.cc", directory: "/tmp/dbginfo")
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = distinct !DISubprogram(name: "a", linkageName: "_Z1ab", line: 17, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 17, file: !1, scope: !5, type: !9, variables: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "bool", size: 8, align: 8, encoding: DW_ATE_boolean)
!12 = !{!13}
!13 = !DILocalVariable(name: "u", line: 17, arg: 1, scope: !8, file: !5, type: !11)
!14 = distinct !DISubprogram(name: "x", linkageName: "_Z1xb", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 5, file: !1, scope: !5, type: !9, variables: !15)
!15 = !{!16, !17}
!16 = !DILocalVariable(name: "b", line: 5, arg: 1, scope: !14, file: !5, type: !11)
!17 = !DILocalVariable(name: "s", line: 7, scope: !18, file: !5, type: !20)
!18 = distinct !DILexicalBlock(line: 6, column: 0, file: !1, scope: !19)
!19 = distinct !DILexicalBlock(line: 6, column: 0, file: !1, scope: !14)
!20 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!21 = !{i32 2, !"Dwarf Version", i32 4}
!22 = !{i32 2, !"Debug Info Version", i32 3}
!23 = !{!"clang version 3.5.0 "}
!24 = !{i1 false}
!25 = !DILocalVariable(name: "b", line: 5, arg: 1, scope: !14, file: !5, type: !11)
!26 = !DILocation(line: 14, scope: !4)
!27 = !DILocation(line: 5, scope: !14, inlinedAt: !26)
!28 = !DILocation(line: 10, scope: !14, inlinedAt: !26)
!29 = !DILocation(line: 15, scope: !4)
!30 = !DILocation(line: 17, scope: !8)
!31 = !DILocalVariable(name: "b", line: 5, arg: 1, scope: !14, file: !5, type: !11)
!32 = !DILocation(line: 18, scope: !8)
!33 = !DILocation(line: 5, scope: !14, inlinedAt: !32)
!34 = !DILocation(line: 6, scope: !19, inlinedAt: !32)
!35 = !DILocation(line: 7, scope: !18, inlinedAt: !32)
!36 = !{!37, !37, i64 0}
!37 = !{!"int", !38, i64 0}
!38 = !{!"omnipotent char", !39, i64 0}
!39 = !{!"Simple C/C++ TBAA"}
!40 = !DILocalVariable(name: "s", line: 7, scope: !18, file: !5, type: !20)
!41 = !DILocation(line: 8, scope: !18, inlinedAt: !32)
!42 = !DILocation(line: 9, scope: !18, inlinedAt: !32)
!43 = !DILocation(line: 10, scope: !14, inlinedAt: !32)
!44 = !DILocation(line: 19, scope: !8)
