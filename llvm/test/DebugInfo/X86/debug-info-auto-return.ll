; RUN: llc %s -filetype=obj -o - | llvm-dwarfdump -v - | FileCheck %s

; CHECK: .debug_info contents:

; CHECK: DW_TAG_subprogram 
; CHECK-NEXT: DW_AT_linkage_name [DW_FORM_strx1]    (indexed {{.*}} string = "_ZN7myClass7findMaxEv")
; CHECK: DW_AT_type [DW_FORM_ref4]     (cu + {{.*}} "auto")
; CHECK-NEXT: DW_AT_declaration [DW_FORM_flag_present]      (true)

; CHECK: DW_TAG_subprogram 
; CHECK: DW_AT_type [DW_FORM_ref4]       (cu + {{.*}} "double")
; CHECK: DW_AT_specification [DW_FORM_ref4]      (cu + {{.*}} "_ZN7myClass7findMaxEv")

; C++ source to regenerate:
; struct myClass {
;    auto findMax();
; };
;
; auto myClass::findMax() {
;    return 0.0;
; }

; $ clang++ -O0 -g -gdwarf-5 debug-info-template-align.cpp -c

; ModuleID = '/dir/test.cpp'
source_filename = "/dir/test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.myClass = type { i8 }
; Function Attrs: noinline nounwind optnone uwtable
define dso_local double @_ZN7myClass7findMaxEv(%struct.myClass* %this) #0 align 2 !dbg !7 {
entry:
  %this.addr = alloca %struct.myClass*, align 8
  store %struct.myClass* %this, %struct.myClass** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.myClass** %this.addr, metadata !17, metadata !DIExpression()), !dbg !19
  %this1 = load %struct.myClass*, %struct.myClass** %this.addr, align 8
  ret double 0.000000e+00, !dbg !20
}
; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone uwtable }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 10.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/dir/test.cpp", directory: "/dir/", checksumkind: CSK_MD5, checksum: "4bed8955bd441e3129c12f557ed53962")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 10.0.0"}
!7 = distinct !DISubprogram(name: "findMax", linkageName: "_ZN7myClass7findMaxEv", scope: !8, file: !1, line: 20, type: !9, scopeLine: 20, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !13, retainedNodes: !2)
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "myClass", file: !1, line: 16, size: 8, flags: DIFlagTypePassByValue, elements: !2, identifier: "_ZTS7myClass")
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !12}
!11 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!13 = !DISubprogram(name: "findMax", linkageName: "_ZN7myClass7findMaxEv", scope: !8, file: !1, line: 17, type: !14, scopeLine: 17, flags: DIFlagPrototyped, spFlags: 0)
!14 = !DISubroutineType(types: !15)
!15 = !{!16, !12}
!16 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "auto")
!17 = !DILocalVariable(name: "this", arg: 1, scope: !7, type: !18, flags: DIFlagArtificial | DIFlagObjectPointer)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!19 = !DILocation(line: 0, scope: !7)
!20 = !DILocation(line: 21, column: 3, scope: !7)
