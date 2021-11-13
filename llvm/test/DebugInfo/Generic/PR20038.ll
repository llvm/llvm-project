; For some reason, the output when targetting sparc is not quite as expected.
; XFAIL: sparc

; RUN: %llc_dwarf -O0 -filetype=obj -dwarf-linkage-names=All < %s | llvm-dwarfdump -debug-info - | FileCheck %s --implicit-check-not=DW_TAG

; IR generated from clang -O0 with:
; struct C {
;   ~C();
; };
; extern bool b;
; void fun4() { b && (C(), 1); }
; __attribute__((always_inline)) C::~C() { }

; CHECK: DW_TAG_compile_unit

; CHECK:   DW_TAG_structure_type
; CHECK:     DW_AT_name ("C")
; CHECK:     DW_TAG_subprogram
; CHECK:       DW_AT_name ("~C")
; CHECK:       DW_TAG_formal_parameter
; CHECK:   DW_TAG_pointer_type

; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_linkage_name ("_ZN1CD1Ev")
; CHECK:     DW_TAG_formal_parameter
; CHECK:       DW_AT_name ("this")
; CHECK:   DW_TAG_pointer_type

; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name ("fun4")
; CHECK:     DW_TAG_inlined_subroutine
; CHECK:       DW_AT_abstract_origin {{.*}} "_ZN1CD1Ev"
; CHECK:       DW_TAG_formal_parameter
; CHECK:         DW_AT_abstract_origin {{.*}} "this"

; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_abstract_origin {{.*}} "_ZN1CD1Ev"
; CHECK:     DW_TAG_formal_parameter
; CHECK:       DW_AT_abstract_origin {{.*}} "this"

; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_linkage_name  ("_ZN1CD2Ev")
; CHECK:     DW_AT_specification {{.*}} "~C"
; CHECK:     DW_TAG_formal_parameter
; CHECK:       DW_AT_name  ("this")

%struct.C = type { i8 }

@b = external global i8

; Function Attrs: nounwind
define void @_Z4fun4v() #0 !dbg !12 {
entry:
  %this.addr.i.i = alloca %struct.C*, align 8, !dbg !21
  %this.addr.i = alloca %struct.C*, align 8, !dbg !22
  %agg.tmp.ensured = alloca %struct.C, align 1
  %cleanup.cond = alloca i1
  %0 = load i8, i8* @b, align 1, !dbg !24
  %tobool = trunc i8 %0 to i1, !dbg !24
  store i1 false, i1* %cleanup.cond
  br i1 %tobool, label %land.rhs, label %land.end, !dbg !24

land.rhs:                                         ; preds = %entry
  store i1 true, i1* %cleanup.cond, !dbg !25
  br label %land.end

land.end:                                         ; preds = %land.rhs, %entry
  %1 = phi i1 [ false, %entry ], [ true, %land.rhs ]
  %cleanup.is_active = load i1, i1* %cleanup.cond, !dbg !27
  br i1 %cleanup.is_active, label %cleanup.action, label %cleanup.done, !dbg !27

cleanup.action:                                   ; preds = %land.end
  store %struct.C* %agg.tmp.ensured, %struct.C** %this.addr.i, align 8, !dbg !22
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr.i, metadata !129, metadata !DIExpression()), !dbg !31
  %this1.i = load %struct.C*, %struct.C** %this.addr.i, !dbg !22
  store %struct.C* %this1.i, %struct.C** %this.addr.i.i, align 8, !dbg !21
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr.i.i, metadata !132, metadata !DIExpression()), !dbg !33
  %this1.i.i = load %struct.C*, %struct.C** %this.addr.i.i, !dbg !21
  br label %cleanup.done, !dbg !22

cleanup.done:                                     ; preds = %cleanup.action, %land.end
  ret void, !dbg !34
}

; Function Attrs: alwaysinline nounwind
define void @_ZN1CD1Ev(%struct.C* %this) unnamed_addr #1 align 2 !dbg !17 {
entry:
  %this.addr.i = alloca %struct.C*, align 8, !dbg !37
  %this.addr = alloca %struct.C*, align 8
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr, metadata !29, metadata !DIExpression()), !dbg !38
  %this1 = load %struct.C*, %struct.C** %this.addr
  store %struct.C* %this1, %struct.C** %this.addr.i, align 8, !dbg !37
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr.i, metadata !232, metadata !DIExpression()), !dbg !39
  %this1.i = load %struct.C*, %struct.C** %this.addr.i, !dbg !37
  ret void, !dbg !37
}

; Function Attrs: alwaysinline nounwind
define void @_ZN1CD2Ev(%struct.C* %this) unnamed_addr #1 align 2 !dbg !16 {
entry:
  %this.addr = alloca %struct.C*, align 8
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.C** %this.addr, metadata !32, metadata !DIExpression()), !dbg !40
  %this1 = load %struct.C*, %struct.C** %this.addr
  ret void, !dbg !41
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { alwaysinline nounwind "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18, !19}
!llvm.ident = !{!20}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "<stdin>", directory: "/tmp/dbginfo")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "C", line: 1, size: 8, align: 8, file: !5, elements: !6, identifier: "_ZTS1C")
!5 = !DIFile(filename: "PR20038.cpp", directory: "/tmp/dbginfo")
!6 = !{!7}
!7 = !DISubprogram(name: "~C", line: 2, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 2, file: !5, scope: !4, type: !8)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !4)
!12 = distinct !DISubprogram(name: "fun4", linkageName: "_Z4fun4v", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 5, file: !5, scope: !13, type: !14, retainedNodes: !2)
!13 = !DIFile(filename: "PR20038.cpp", directory: "/tmp/dbginfo")
!14 = !DISubroutineType(types: !15)
!15 = !{null}
!16 = distinct !DISubprogram(name: "~C", linkageName: "_ZN1CD2Ev", line: 6, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 6, file: !5, scope: !4, type: !8, declaration: !7, retainedNodes: !2)
!17 = distinct !DISubprogram(name: "~C", linkageName: "_ZN1CD1Ev", line: 6, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 6, file: !5, scope: !4, type: !8, declaration: !7, retainedNodes: !2)
!18 = !{i32 2, !"Dwarf Version", i32 4}
!19 = !{i32 2, !"Debug Info Version", i32 3}
!20 = !{!"clang version 3.5.0 "}
!21 = !DILocation(line: 6, scope: !17, inlinedAt: !22)
!22 = !DILocation(line: 5, scope: !23)
!23 = distinct !DILexicalBlock(line: 5, column: 0, file: !5, scope: !12)
!24 = !DILocation(line: 5, scope: !12)
!25 = !DILocation(line: 5, scope: !26)
!26 = distinct !DILexicalBlock(line: 5, column: 0, file: !5, scope: !12)
!27 = !DILocation(line: 5, scope: !28)
!28 = distinct !DILexicalBlock(line: 5, column: 0, file: !5, scope: !12)
!29 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !17, type: !30)
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !4)
!31 = !DILocation(line: 0, scope: !17, inlinedAt: !22)
!32 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !16, type: !30)
!33 = !DILocation(line: 0, scope: !16, inlinedAt: !21)

!129 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !17, type: !30)
!132 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !16, type: !30)
!232 = !DILocalVariable(name: "this", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !16, type: !30)

!34 = !DILocation(line: 5, scope: !35)
!35 = distinct !DILexicalBlock(line: 5, column: 0, file: !5, scope: !36)
!36 = distinct !DILexicalBlock(line: 5, column: 0, file: !5, scope: !12)
!37 = !DILocation(line: 6, scope: !17)
!38 = !DILocation(line: 0, scope: !17)
!39 = !DILocation(line: 0, scope: !16, inlinedAt: !37)
!40 = !DILocation(line: 0, scope: !16)
!41 = !DILocation(line: 6, scope: !16)
