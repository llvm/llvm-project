; RUN: llc -filetype=obj %s -o - | llvm-dwarfdump -  | FileCheck %s

; Test that byval args get correct DWARF debug locations.
; These end up in the callee as Wasm locals, which is atypical.

; From C code:

; typedef union {	int x; float y; } u;
; typedef struct { int x;	float y; } s;
; int some_func(int x, u some_union, s some_struct, int a[3]) {
;     return x + some_union.x + some_struct.x + a[0];
; }


; ModuleID = 't.c'
source_filename = "t.c"
target triple = "wasm32-unknown-unknown"

%union.u = type { i32 }
%struct.s = type { i32, float }

; Function Attrs: noinline nounwind optnone
define hidden i32 @some_func(i32 %x, %union.u* byval(%union.u) align 4 %some_union, %struct.s* byval(%struct.s) align 4 %some_struct, i32* %a) #0 !dbg !7 {
entry:
  %x.addr = alloca i32, align 4
  %a.addr = alloca i32*, align 4
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %x.addr, metadata !23, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.declare(metadata %union.u* %some_union, metadata !25, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.declare(metadata %struct.s* %some_struct, metadata !27, metadata !DIExpression()), !dbg !28
  store i32* %a, i32** %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32** %a.addr, metadata !29, metadata !DIExpression()), !dbg !30
  %0 = load i32, i32* %x.addr, align 4, !dbg !31
  %x1 = bitcast %union.u* %some_union to i32*, !dbg !32
  %1 = load i32, i32* %x1, align 4, !dbg !32
  %add = add nsw i32 %0, %1, !dbg !33
  %x2 = getelementptr inbounds %struct.s, %struct.s* %some_struct, i32 0, i32 0, !dbg !34
  %2 = load i32, i32* %x2, align 4, !dbg !34
  %add3 = add nsw i32 %add, %2, !dbg !35
  %3 = load i32*, i32** %a.addr, align 4, !dbg !36
  %arrayidx = getelementptr inbounds i32, i32* %3, i32 0, !dbg !36
  %4 = load i32, i32* %arrayidx, align 4, !dbg !36
  %add4 = add nsw i32 %add3, %4, !dbg !37
  ret i32 %add4, !dbg !38
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0 (https://github.com/llvm/llvm-project.git eccc734a69c0c012ae3160887b65a535b35ead3e)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "C:\\src\\llvm")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 12.0.0 (https://github.com/llvm/llvm-project.git eccc734a69c0c012ae3160887b65a535b35ead3e)"}
!7 = distinct !DISubprogram(name: "some_func", scope: !1, file: !1, line: 11, type: !8, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10, !11, !17, !22}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "u", file: !1, line: 4, baseType: !12)
!12 = distinct !DICompositeType(tag: DW_TAG_union_type, file: !1, line: 1, size: 32, elements: !13)
!13 = !{!14, !15}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !12, file: !1, line: 2, baseType: !10, size: 32)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !12, file: !1, line: 3, baseType: !16, size: 32)
!16 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!17 = !DIDerivedType(tag: DW_TAG_typedef, name: "s", file: !1, line: 9, baseType: !18)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !1, line: 6, size: 64, elements: !19)
!19 = !{!20, !21}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !18, file: !1, line: 7, baseType: !10, size: 32)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !18, file: !1, line: 8, baseType: !16, size: 32, offset: 32)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 32)
!23 = !DILocalVariable(name: "x", arg: 1, scope: !7, file: !1, line: 11, type: !10)
!24 = !DILocation(line: 11, column: 19, scope: !7)
!25 = !DILocalVariable(name: "some_union", arg: 2, scope: !7, file: !1, line: 11, type: !11)
!26 = !DILocation(line: 11, column: 24, scope: !7)
!27 = !DILocalVariable(name: "some_struct", arg: 3, scope: !7, file: !1, line: 11, type: !17)
!28 = !DILocation(line: 11, column: 38, scope: !7)
!29 = !DILocalVariable(name: "a", arg: 4, scope: !7, file: !1, line: 11, type: !22)
!30 = !DILocation(line: 11, column: 55, scope: !7)
!31 = !DILocation(line: 12, column: 12, scope: !7)
!32 = !DILocation(line: 12, column: 27, scope: !7)
!33 = !DILocation(line: 12, column: 14, scope: !7)
!34 = !DILocation(line: 12, column: 43, scope: !7)
!35 = !DILocation(line: 12, column: 29, scope: !7)
!36 = !DILocation(line: 12, column: 47, scope: !7)
!37 = !DILocation(line: 12, column: 45, scope: !7)
!38 = !DILocation(line: 12, column: 5, scope: !7)




; CHECK-LABEL:  DW_TAG_formal_parameter
; CHECK-NEXT:     DW_AT_location        (DW_OP_fbreg +12)
; CHECK-NEXT:     DW_AT_name    ("x")

; CHECK-LABEL:  DW_TAG_formal_parameter
; CHECK-NEXT:     DW_AT_location        (DW_OP_WASM_location 0x0 0x1)
; CHECK-NEXT:     DW_AT_name    ("some_union")

; CHECK-LABEL:  DW_TAG_formal_parameter
; CHECK-NEXT:     DW_AT_location        (DW_OP_WASM_location 0x0 0x2)
; CHECK-NEXT:     DW_AT_name    ("some_struct")

; CHECK-LABEL:  DW_TAG_formal_parameter
; CHECK-NEXT:     DW_AT_location        (DW_OP_fbreg +8)
; CHECK-NEXT:     DW_AT_name    ("a")
