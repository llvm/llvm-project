; RUN: %clang_cc1 -fno-eliminate-unused-debug-types -emit-llvm -o - %s | FileCheck %s
; RUN: llvm-dwarfdump %t | FileCheck %s

;CHECK:   DW_TAG_structure_type
;CHECK-NEXT:                DW_AT_calling_convention        (DW_CC_pass_by_value)
;CHECK-NEXT:                DW_AT_name      ("Type")
;CHECK-NEXT:                DW_AT_byte_size (0x04)
;CHECK-NEXT:                DW_AT_decl_file ("/iusers/ykhatavk/project_ics/llvm-project/debug_issue/enump.cpp")
;CHECK-NEXT:                DW_AT_decl_line (1)
;CHECK:     DW_TAG_member
;CHECK-NEXT:                  DW_AT_name    ("value")
;CHECK-NEXT:                  DW_AT_type    (0x0000004f "int")
;CHECK-NEXT:                  DW_AT_decl_file       ("/iusers/ykhatavk/project_ics/llvm-project/debug_issue/enump.cpp")
;CHECK-NEXT:                  DW_AT_decl_line       (3)
;CHECK-NEXT:                  DW_AT_data_member_location    (0x00)
;CHECK:     DW_TAG_enumeration_type
;CHECK-NEXT:                  DW_AT_type    (0x00000053 "unsigned int")
;CHECK-NEXT:                  DW_AT_byte_size       (0x04)
;CHECK-NEXT:                  DW_AT_decl_file       ("/iusers/ykhatavk/project_ics/llvm-project/debug_issue/enump.cpp")
;CHECK-NEXT:                  DW_AT_decl_line       (2)
;CHECK:       DW_TAG_enumerator
;CHECK-NEXT:                    DW_AT_name  ("Unused")
;CHECK-NEXT:                    DW_AT_const_value   (0)

; ModuleID = 'enump.cpp'
source_filename = "enump.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Type = type { i32 }

$_ZN4TypeC2Ev = comdat any

; Function Attrs: mustprogress noinline norecurse nounwind optnone uwtable
define dso_local noundef i32 @main() #0 !dbg !20 {
entry:
  %retval = alloca i32, align 4
  %t = alloca %struct.Type, align 4
  store i32 0, ptr %retval, align 4
    #dbg_declare(ptr %t, !24, !DIExpression(), !25)
  call void @_ZN4TypeC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %t) #2, !dbg !25
  %value = getelementptr inbounds nuw %struct.Type, ptr %t, i32 0, i32 0, !dbg !26
  %0 = load i32, ptr %value, align 4, !dbg !26
  ret i32 %0, !dbg !27
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4TypeC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this) unnamed_addr #1 comdat align 2 !dbg !28 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !33, !DIExpression(), !35)
  %this1 = load ptr, ptr %this.addr, align 8
  %value = getelementptr inbounds nuw %struct.Type, ptr %this1, i32 0, i32 0, !dbg !36
  store i32 0, ptr %value, align 4, !dbg !36
  ret void, !dbg !37
}

attributes #0 = { mustprogress noinline norecurse nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13, !14, !15, !16, !17, !18}
!llvm.ident = !{!19}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 21.0.0git (https://github.com/llvm/llvm-project.git 2b983a24583dd4e131d727717872a56712b5dd52)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !11, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "enump.cpp", directory: "/iusers/ykhatavk/project_ics/llvm-project/debug_issue", checksumkind: CSK_MD5, checksum: "35ec0482db78c9e464e8bf8cd231d5b9")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, scope: !4, file: !1, line: 2, baseType: !8, size: 32, elements: !9, identifier: "_ZTSN4TypeUt_E")
!4 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Type", file: !1, line: 1, size: 32, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !5, identifier: "_ZTS4Type")
!5 = !{!3, !6}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !4, file: !1, line: 3, baseType: !7, size: 32)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!9 = !{!10}
!10 = !DIEnumerator(name: "Unused", value: 0, isUnsigned: true)
!11 = !{!4}
!12 = !{i32 7, !"Dwarf Version", i32 5}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{i32 8, !"PIC Level", i32 2}
!16 = !{i32 7, !"PIE Level", i32 2}
!17 = !{i32 7, !"uwtable", i32 2}
!18 = !{i32 7, !"frame-pointer", i32 2}
!19 = !{!"clang version 21.0.0git (https://github.com/llvm/llvm-project.git 2b983a24583dd4e131d727717872a56712b5dd52)"}
!20 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 5, type: !21, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !23)
!21 = !DISubroutineType(types: !22)
!22 = !{!7}
!23 = !{}
!24 = !DILocalVariable(name: "t", scope: !20, file: !1, line: 6, type: !4)
!25 = !DILocation(line: 6, column: 10, scope: !20)
!26 = !DILocation(line: 7, column: 14, scope: !20)
!27 = !DILocation(line: 7, column: 5, scope: !20)
!28 = distinct !DISubprogram(name: "Type", linkageName: "_ZN4TypeC2Ev", scope: !4, file: !1, line: 1, type: !29, scopeLine: 1, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !32, retainedNodes: !23)
!29 = !DISubroutineType(types: !30)
!30 = !{null, !31}
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!32 = !DISubprogram(name: "Type", scope: !4, type: !29, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: 0)
!33 = !DILocalVariable(name: "this", arg: 1, scope: !28, type: !34, flags: DIFlagArtificial | DIFlagObjectPointer)
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64)
!35 = !DILocation(line: 0, scope: !28)
!36 = !DILocation(line: 3, column: 9, scope: !28)
!37 = !DILocation(line: 1, column: 8, scope: !28)
