; RUN: llc  -mtriple=x86_64-unknown-linux-gnu %s -o %t -filetype=obj
; RUN: llvm-dwarfdump %t | FileCheck %s
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

; CHECK: DW_TAG_structure_type
; CHECK:      DW_AT_name	("A<int>")
; CHECK-NEXT:  DW_AT_byte_size	(0x08)
; CHECK-NEXT:  DW_AT_decl_file	("test.cpp")
; CHECK-NEXT:  DW_AT_decl_line	(2)

; CHECK-NOT: NULL

; CHECK:[[TEMPLATE:0x[0-9a-f]*]]: DW_TAG_template_type_parameter
; CHECK-NEXT: DW_AT_type      {{.*}} "int"
; CHECK-NEXT: DW_AT_name	("T")

; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name  ("val_")
; CHECK-NEXT: DW_AT_type ([[TEMPLATE]] "T")

; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name	("val2_")
; CHECK-NEXT: DW_AT_type	{{.*}} "int"
; ModuleID = 'test.cpp'
source_filename = "test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { i32, i32 }

$_ZN1AIiEC2Ev = comdat any

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @main() #0 !dbg !22 {
entry:
  %retval = alloca i32, align 4
  %a = alloca %struct.A, align 4
  store i32 0, ptr %retval, align 4
    #dbg_declare(ptr %a, !26, !DIExpression(), !27)
  call void @_ZN1AIiEC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %a), !dbg !27
  %val2_ = getelementptr inbounds nuw %struct.A, ptr %a, i32 0, i32 1, !dbg !28
  store i32 3, ptr %val2_, align 4, !dbg !29
  ret i32 0, !dbg !30
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN1AIiEC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %this) unnamed_addr #1 comdat align 2 !dbg !31 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !32, !DIExpression(), !34)
  %this1 = load ptr, ptr %this.addr, align 8
  %val_ = getelementptr inbounds nuw %struct.A, ptr %this1, i32 0, i32 0, !dbg !35
  store i32 0, ptr %val_, align 4, !dbg !35
  %val2_ = getelementptr inbounds nuw %struct.A, ptr %this1, i32 0, i32 1, !dbg !36
  store i32 0, ptr %val2_, align 4, !dbg !36
  ret void, !dbg !37
}

attributes #0 = { mustprogress noinline norecurse optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !15, !16, !17, !18, !19, !20}
!llvm.ident = !{!21}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "", checksumkind: CSK_MD5, checksum: "451371997e00e9e85d610a4e9d44a9b5")
!2 = !{!3}
!3 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A<int>", file: !1, line: 2, size: 64, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !4, templateParams: !12, identifier: "_ZTS1AIiE")
!4 = !{!5, !7, !8}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "val_", scope: !3, file: !1, line: 4, baseType: !38, size: 32)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "val2_", scope: !3, file: !1, line: 5, baseType: !6, size: 32, offset: 32)
!8 = !DISubprogram(name: "A", scope: !3, file: !1, line: 3, type: !9, scopeLine: 3, flags: DIFlagPrototyped, spFlags: 0)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!12 = !{!38}
!13 = !DITemplateTypeParameter(name: "T", type: !6)
!14 = !{i32 7, !"Dwarf Version", i32 5}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{i32 8, !"PIC Level", i32 2}
!18 = !{i32 7, !"PIE Level", i32 2}
!19 = !{i32 7, !"uwtable", i32 2}
!20 = !{i32 7, !"frame-pointer", i32 2}
!21 = !{!"clang"}
!22 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 7, type: !23, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !25)
!23 = !DISubroutineType(types: !24)
!24 = !{!6}
!25 = !{}
!26 = !DILocalVariable(name: "a", scope: !22, file: !1, line: 8, type: !3)
!27 = !DILocation(line: 8, column: 8, scope: !22)
!28 = !DILocation(line: 9, column: 3, scope: !22)
!29 = !DILocation(line: 9, column: 9, scope: !22)
!30 = !DILocation(line: 10, column: 1, scope: !22)
!31 = distinct !DISubprogram(name: "A", linkageName: "_ZN1AIiEC2Ev", scope: !3, file: !1, line: 3, type: !9, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !8, retainedNodes: !25)
!32 = !DILocalVariable(name: "this", arg: 1, scope: !31, type: !33, flags: DIFlagArtificial | DIFlagObjectPointer)
!33 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64)
!34 = !DILocation(line: 0, scope: !31)
!35 = !DILocation(line: 3, column: 8, scope: !31)
!36 = !DILocation(line: 3, column: 17, scope: !31)
!37 = !DILocation(line: 3, column: 28, scope: !31)
!38 = !DIDerivedType(tag: DW_TAG_template_type_parameter,name:"T",scope:!3, baseType: !6)
