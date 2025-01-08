; RUN: llc < %s -filetype=obj -o %t.o
; RUN: llvm-pdbutil dump -types %t.o | FileCheck %s

; RUN: llc --try-experimental-debuginfo-iterators < %s -filetype=obj -o %t.o
; RUN: llvm-pdbutil dump -types %t.o | FileCheck %s

; C++ source:
; // Note that MSVC doesn't emit anything about WrapTypedef or WrapTypedef::Inner!
; struct WrapTypedef {
;   typedef int Inner;
; };
; struct WrapStruct {
;   struct Inner { int x; };
; };
; struct WrapClass {
;   class Inner { public: int x; };
; };
; struct WrapEnum {
;   enum Inner { One, Two };
; };
; struct WrapUnion {
;   union Inner { int x; float y; };
; };
; void useInnerTypes() {
;   WrapTypedef::Inner v1;
;   WrapStruct::Inner v2;
;   WrapClass::Inner v3;
;   WrapEnum::Inner v4;
;   WrapUnion::Inner v5;
; }

; There should be two LF_STRUCTURE records for each wrapped type, forward decl
; and complete type. For every inner record type, there should be two. Enums
; don't get forward decls.

; CHECK-DAG: | LF_STRUCTURE {{.*}} `WrapStruct`
; CHECK-DAG: | LF_STRUCTURE {{.*}} `WrapStruct`
; CHECK-DAG: | LF_STRUCTURE {{.*}} `WrapStruct::Inner`
; CHECK-DAG: | LF_STRUCTURE {{.*}} `WrapStruct::Inner`
; CHECK-DAG: | LF_STRUCTURE {{.*}} `WrapClass`
; CHECK-DAG: | LF_STRUCTURE {{.*}} `WrapClass`
; CHECK-DAG: | LF_CLASS {{.*}} `WrapClass::Inner`
; CHECK-DAG: | LF_CLASS {{.*}} `WrapClass::Inner`
; CHECK-DAG: | LF_STRUCTURE {{.*}} `WrapEnum`
; CHECK-DAG: | LF_STRUCTURE {{.*}} `WrapEnum`
; CHECK-DAG: | LF_ENUM {{.*}} `WrapEnum::Inner`
; CHECK-DAG: | LF_STRUCTURE {{.*}} `WrapUnion`
; CHECK-DAG: | LF_STRUCTURE {{.*}} `WrapUnion`
; CHECK-DAG: | LF_UNION {{.*}} `WrapUnion::Inner`
; CHECK-DAG: | LF_UNION {{.*}} `WrapUnion::Inner`

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.23.28106"

%"struct.WrapStruct::Inner" = type { i32 }
%"class.WrapClass::Inner" = type { i32 }
%"union.WrapUnion::Inner" = type { i32 }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?useInnerTypes@@YAXXZ"() #0 !dbg !15 {
entry:
  %v1 = alloca i32, align 4
  %v2 = alloca %"struct.WrapStruct::Inner", align 4
  %v3 = alloca %"class.WrapClass::Inner", align 4
  %v4 = alloca i32, align 4
  %v5 = alloca %"union.WrapUnion::Inner", align 4
  call void @llvm.dbg.declare(metadata ptr %v1, metadata !19, metadata !DIExpression()), !dbg !23
  call void @llvm.dbg.declare(metadata ptr %v2, metadata !24, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.declare(metadata ptr %v3, metadata !31, metadata !DIExpression()), !dbg !37
  call void @llvm.dbg.declare(metadata ptr %v4, metadata !38, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata ptr %v5, metadata !40, metadata !DIExpression()), !dbg !48
  ret void, !dbg !49
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11, !12, !13}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 10.0.0 (git@github.com:llvm/llvm-project.git a8ccb48f697d3fbe85c593248ff1053fdf522a6e)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "t.cpp", directory: "C:\\src\\llvm-project\\build", checksumkind: CSK_MD5, checksum: "4228f12f516cd3d6dd76462be09ec111")
!2 = !{!3, !3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Inner", scope: !4, file: !1, line: 11, baseType: !6, size: 32, elements: !7, identifier: ".?AW4Inner@WrapEnum@@")
!4 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "WrapEnum", file: !1, line: 10, size: 8, flags: DIFlagTypePassByValue, elements: !5, identifier: ".?AUWrapEnum@@")
!5 = !{!3}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{!8, !9}
!8 = !DIEnumerator(name: "One", value: 0)
!9 = !DIEnumerator(name: "Two", value: 1)
!10 = !{i32 2, !"CodeView", i32 1}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 2}
!13 = !{i32 7, !"PIC Level", i32 2}
!14 = !{!"clang version 10.0.0 (git@github.com:llvm/llvm-project.git a8ccb48f697d3fbe85c593248ff1053fdf522a6e)"}
!15 = distinct !DISubprogram(name: "useInnerTypes", linkageName: "?useInnerTypes@@YAXXZ", scope: !1, file: !1, line: 16, type: !16, scopeLine: 16, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !18)
!16 = !DISubroutineType(types: !17)
!17 = !{null}
!18 = !{}
!19 = !DILocalVariable(name: "v1", scope: !15, file: !1, line: 17, type: !20)
!20 = !DIDerivedType(tag: DW_TAG_typedef, name: "Inner", scope: !21, file: !1, line: 2, baseType: !6)
!21 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "WrapTypedef", file: !1, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !22, identifier: ".?AUWrapTypedef@@")
!22 = !{!20}
!23 = !DILocation(line: 17, scope: !15)
!24 = !DILocalVariable(name: "v2", scope: !15, file: !1, line: 18, type: !25)
!25 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Inner", scope: !26, file: !1, line: 5, size: 32, flags: DIFlagTypePassByValue, elements: !28, identifier: ".?AUInner@WrapStruct@@")
!26 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "WrapStruct", file: !1, line: 4, size: 8, flags: DIFlagTypePassByValue, elements: !27, identifier: ".?AUWrapStruct@@")
!27 = !{!25}
!28 = !{!29}
!29 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !25, file: !1, line: 5, baseType: !6, size: 32)
!30 = !DILocation(line: 18, scope: !15)
!31 = !DILocalVariable(name: "v3", scope: !15, file: !1, line: 19, type: !32)
!32 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "Inner", scope: !33, file: !1, line: 8, size: 32, flags: DIFlagTypePassByValue, elements: !35, identifier: ".?AVInner@WrapClass@@")
!33 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "WrapClass", file: !1, line: 7, size: 8, flags: DIFlagTypePassByValue, elements: !34, identifier: ".?AUWrapClass@@")
!34 = !{!32}
!35 = !{!36}
!36 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !32, file: !1, line: 8, baseType: !6, size: 32, flags: DIFlagPublic)
!37 = !DILocation(line: 19, scope: !15)
!38 = !DILocalVariable(name: "v4", scope: !15, file: !1, line: 20, type: !3)
!39 = !DILocation(line: 20, scope: !15)
!40 = !DILocalVariable(name: "v5", scope: !15, file: !1, line: 21, type: !41)
!41 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "Inner", scope: !42, file: !1, line: 14, size: 32, flags: DIFlagTypePassByValue, elements: !44, identifier: ".?ATInner@WrapUnion@@")
!42 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "WrapUnion", file: !1, line: 13, size: 8, flags: DIFlagTypePassByValue, elements: !43, identifier: ".?AUWrapUnion@@")
!43 = !{!41}
!44 = !{!45, !46}
!45 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !41, file: !1, line: 14, baseType: !6, size: 32)
!46 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !41, file: !1, line: 14, baseType: !47, size: 32)
!47 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!48 = !DILocation(line: 21, scope: !15)
!49 = !DILocation(line: 22, scope: !15)
