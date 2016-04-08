; RUN: llc %s -filetype=obj -o %t.o
; RUN: llvm-dwarfdump -debug-dump=info %t.o | FileCheck %s
; Test emitting a constant for an aggregate type.
;
; clang -S -O1 -emit-llvm
;
; typedef struct { unsigned i; } S;
;
; unsigned foo(S s) {
;   s.i = 1;
;   return s.i;
; }
;
; class C { public: unsigned i; };
;
; unsigned foo(C c) {
;   c.i = 2;
;   return c.i;
; }
;
; unsigned bar() {
;  int a[1] = { 3 };
;   return a[0];
; }
;
; CHECK:  DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_const_value [DW_FORM_udata]	(1)
; CHECK-NEXT: DW_AT_name {{.*}} "s"
;
; CHECK:  DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_const_value [DW_FORM_udata]	(2)
; CHECK-NEXT: DW_AT_name {{.*}} "c"
;
; CHECK:  DW_TAG_variable
; CHECK-NEXT: DW_AT_const_value [DW_FORM_udata]	(3)
; CHECK-NEXT: DW_AT_name {{.*}} "a"

; ModuleID = 'sroasplit-4.cpp'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; Function Attrs: nounwind readnone ssp uwtable
define i32 @_Z3foo1S(i32 %s.coerce) #0 !dbg !12 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %s.coerce, i64 0, metadata !18, metadata !37), !dbg !38
  tail call void @llvm.dbg.value(metadata i32 1, i64 0, metadata !18, metadata !37), !dbg !38
  ret i32 1, !dbg !39
}

; Function Attrs: nounwind readnone ssp uwtable
define i32 @_Z3foo1C(i32 %c.coerce) #0 !dbg !19 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %c.coerce, i64 0, metadata !23, metadata !37), !dbg !40
  tail call void @llvm.dbg.value(metadata i32 2, i64 0, metadata !23, metadata !37), !dbg !40
  ret i32 2, !dbg !41
}

; Function Attrs: nounwind readnone ssp uwtable
define i32 @_Z3barv() #0 !dbg !24 {
entry:
  tail call void @llvm.dbg.value(metadata i32 3, i64 0, metadata !28, metadata !37), !dbg !42
  ret i32 3, !dbg !43
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind readnone ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!33, !34, !35}
!llvm.ident = !{!36}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.6.0 (trunk 225364) (llvm/trunk 225366)", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, subprograms: !11, globals: !2, imports: !2)
!1 = !DIFile(filename: "sroasplit-4.cpp", directory: "")
!2 = !{}
!3 = !{!4, !8}
!4 = !DICompositeType(tag: DW_TAG_structure_type, line: 1, size: 32, align: 32, file: !1, elements: !5, identifier: "_ZTS1S")
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "i", line: 1, size: 32, align: 32, file: !1, scope: !"_ZTS1S", baseType: !7)
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!8 = !DICompositeType(tag: DW_TAG_class_type, name: "C", line: 8, size: 32, align: 32, file: !1, elements: !9, identifier: "_ZTS1C")
!9 = !{!10}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "i", line: 8, size: 32, align: 32, flags: DIFlagPublic, file: !1, scope: !"_ZTS1C", baseType: !7)
!11 = !{!12, !19, !24}
!12 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foo1S", line: 3, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 3, file: !1, scope: !13, type: !14, variables: !17)
!13 = !DIFile(filename: "sroasplit-4.cpp", directory: "")
!14 = !DISubroutineType(types: !15)
!15 = !{!7, !16}
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "S", line: 1, file: !1, baseType: !"_ZTS1S")
!17 = !{!18}
!18 = !DILocalVariable(name: "s", line: 3, arg: 1, scope: !12, file: !13, type: !16)
!19 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foo1C", line: 10, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 10, file: !1, scope: !13, type: !20, variables: !22)
!20 = !DISubroutineType(types: !21)
!21 = !{!7, !"_ZTS1C"}
!22 = !{!23}
!23 = !DILocalVariable(name: "c", line: 10, arg: 1, scope: !19, file: !13, type: !"_ZTS1C")
!24 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", line: 15, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 15, file: !1, scope: !13, type: !25, variables: !27)
!25 = !DISubroutineType(types: !26)
!26 = !{!7}
!27 = !{!28}
!28 = !DILocalVariable(name: "a", line: 16, scope: !24, file: !13, type: !29)
!29 = !DICompositeType(tag: DW_TAG_array_type, size: 32, align: 32, baseType: !30, elements: !31)
!30 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!31 = !{!32}
!32 = !DISubrange(count: 1)
!33 = !{i32 2, !"Dwarf Version", i32 2}
!34 = !{i32 2, !"Debug Info Version", i32 3}
!35 = !{i32 1, !"PIC Level", i32 2}
!36 = !{!"clang version 3.6.0 (trunk 225364) (llvm/trunk 225366)"}
!37 = !DIExpression()
!38 = !DILocation(line: 3, column: 16, scope: !12)
!39 = !DILocation(line: 5, column: 3, scope: !12)
!40 = !DILocation(line: 10, column: 16, scope: !19)
!41 = !DILocation(line: 12, column: 3, scope: !19)
!42 = !DILocation(line: 16, column: 6, scope: !24)
!43 = !DILocation(line: 17, column: 3, scope: !24)
