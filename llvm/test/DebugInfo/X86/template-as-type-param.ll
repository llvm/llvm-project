; RUN: llc  -mtriple=x86_64-unknown-linux-gnu %s -o %t -filetype=obj
; RUN:llvm-dwarfdump %t | FileCheck %s
;Source code for the IR below:
;template <typename T>
;struct A
;{
;  A () : val_ (), val2_ () { }
;  T val_;
;  int val2_;
;};

;int main (void)
;{
; A<int> a;
;  a.val2_ = 3;
; return 0;
;}

;CHECK: DW_TAG_structure_type
;CHECK:      DW_AT_name	("A<int>")
;CHECK-NEXT:  DW_AT_byte_size	(0x08)
;CHECK-NEXT:  DW_AT_decl_file	("/path/to{{/|\\}}test.cpp")
;CHECK-NEXT:  DW_AT_decl_line	(2)

;CHECK-NOT: NULL

;CHECK:[[TEMPLATE:0x[0-9a-f]*]]: DW_TAG_template_type_parameter
;CHECK-NEXT: DW_AT_type      {{.*}} "int"
;CHECK-NEXT: DW_AT_name	("T")

;CHECK: DW_TAG_member
;CHECK-NEXT: DW_AT_name  ("val_")
;CHECK-NEXT: DW_AT_type ([[TEMPLATE]] "T")

;CHECK: DW_TAG_member
;CHECK-NEXT: DW_AT_name	("val2_")
;CHECK-NEXT: DW_AT_type	{{.*}} "int"


; ModuleID = '<stdin>'
source_filename = "test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { i32, i32 }

$_ZN1AIiEC2Ev = comdat any

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @main() #0 !dbg !20 {
entry:
  %retval = alloca i32, align 4
  %a = alloca %struct.A, align 4
  store i32 0, i32* %retval, align 4
  call void @llvm.dbg.declare(metadata %struct.A* %a, metadata !24, metadata !DIExpression()), !dbg !25
  call void @_ZN1AIiEC2Ev(%struct.A* noundef nonnull align 4 dereferenceable(8) %a), !dbg !25
  %val2_ = getelementptr inbounds %struct.A, %struct.A* %a, i32 0, i32 1, !dbg !26
  store i32 3, i32* %val2_, align 4, !dbg !27
  ret i32 0, !dbg !28
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN1AIiEC2Ev(%struct.A* noundef nonnull align 4 dereferenceable(8) %this) unnamed_addr #2 comdat align 2 !dbg !29 {

entry:
  %this.addr = alloca %struct.A*, align 8
  store %struct.A* %this, %struct.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.A** %this.addr, metadata !30, metadata !DIExpression()), !dbg !32
  %this1 = load %struct.A*, %struct.A** %this.addr, align 8
  %val_ = getelementptr inbounds %struct.A, %struct.A* %this1, i32 0, i32 0, !dbg !33
  store i32 0, i32* %val_, align 4, !dbg !33
  %val2_ = getelementptr inbounds %struct.A, %struct.A* %this1, i32 0, i32 1, !dbg !34
  store i32 0, i32* %val2_, align 4, !dbg !34
  ret void, !dbg !35
}

attributes #0 = { mustprogress noinline norecurse optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !15, !16, !17, !18}
!llvm.ident = !{!19}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "", isOptimized: false, flags: " -S -g -O0 -emit-llvm reproducer.cxx -o reproducer.ll", runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/path/to")
!2 = !{!3}
!3 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A<int>", file: !1, line: 2, size: 64, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !4, templateParams: !12, identifier: "_ZTS1AIiE")
!4 = !{!5, !7, !8}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "val_", scope: !3, file: !1, line: 5, baseType: !36, size: 32,offset: 32)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "val2_", scope: !3, file: !1, line: 6, baseType: !6, size: 32, offset: 32)
!8 = !DISubprogram(name: "A", scope: !3, file: !1, line: 4, type: !9, scopeLine: 4, flags: DIFlagPrototyped, spFlags: 0)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!12 = !{!36}
!13 = !DITemplateTypeParameter(name: "T", type: !6)
!14 = !{i32 7, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{i32 7, !"uwtable", i32 2}
!18 = !{i32 7, !"frame-pointer", i32 2}
!19 = !{!""}
!20 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 10, type: !21, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !23)
!21 = !DISubroutineType(types: !22)
!22 = !{!6}
!23 = !{}
!24 = !DILocalVariable(name: "a", scope: !20, file: !1, line: 12, type: !3)
!25 = !DILocation(line: 12, column: 10, scope: !20)
!26 = !DILocation(line: 13, column: 5, scope: !20)
!27 = !DILocation(line: 13, column: 11, scope: !20)
!28 = !DILocation(line: 15, column: 3, scope: !20)
!29 = distinct !DISubprogram(name: "A", linkageName: "_ZN1AIiEC2Ev", scope: !3, file: !1, line: 4, type: !9, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !8, retainedNodes: !23)
!30 = !DILocalVariable(name: "this", arg: 1, scope: !29, type: !31, flags: DIFlagArtificial | DIFlagObjectPointer)
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64)
!32 = !DILocation(line: 0, scope: !29)
!33 = !DILocation(line: 4, column: 10, scope: !29)
!34 = !DILocation(line: 4, column: 19, scope: !29)
!35 = !DILocation(line: 4, column: 30, scope: !29)
!36 = !DIDerivedType(tag: DW_TAG_template_type_parameter,name:"T",scope:!3, baseType: !6)
