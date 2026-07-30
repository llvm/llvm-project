; RUN: llc < %s -filetype=obj -o %t
; RUN: llvm-dwarfdump -v %t | FileCheck %s

; C++ source to regenerate:
; class deleted {
; public:
;   // Defaulted on purpose, so as to facilitate object creation
;    deleted() = default;
; 
;   deleted(const deleted &) = delete;
;   deleted &operator=(const deleted &) = delete;
; 
;   deleted(deleted &&) = delete;
;   deleted &operator=(deleted &&) = delete;
;
;   void func() && = delete;
;   static void bar() = delete;
; 
;   ~deleted() = default;
; };
; 
; void foo() {
;   deleted obj1;
; }
; $ clang++ -O0 -g -gdwarf-5 debug-info-deleted.cpp -c


; CHECK: .debug_abbrev contents:

; CHECK: [7] DW_TAG_subprogram   DW_CHILDREN_yes
; CHECK: DW_AT_deleted   DW_FORM_flag_present
; CHECK: [9] DW_TAG_subprogram   DW_CHILDREN_yes
; CHECK: DW_AT_deleted   DW_FORM_flag_present

; CHECK: .debug_info contents:

; CHECK: DW_TAG_subprogram [7]
; CHECK-NEXT: DW_AT_name [DW_FORM_strx1]    (indexed (00000006) string = "deleted") 
; CHECK:  DW_AT_deleted [DW_FORM_flag_present]  (true)

; CHECK: DW_TAG_subprogram [9]
; CHECK-NEXT: DW_AT_linkage_name [DW_FORM_strx1]    (indexed (00000007) string = "_ZN7deletedaSERKS_") 
; CHECK:  DW_AT_deleted [DW_FORM_flag_present]  (true)

; CHECK: DW_TAG_subprogram [7]
; CHECK-NEXT: DW_AT_name [DW_FORM_strx1]    (indexed (00000006) string = "deleted") 
; CHECK:  DW_AT_deleted [DW_FORM_flag_present]  (true)

; CHECK: DW_TAG_subprogram [9]
; CHECK-NEXT: DW_AT_linkage_name [DW_FORM_strx1]    (indexed (00000009) string = "_ZN7deletedaSEOS_")
; CHECK-NEXT: DW_AT_name [DW_FORM_strx1]    (indexed (00000008) string = "operator=")
; CHECK:  DW_AT_deleted [DW_FORM_flag_present]  (true)

; CHECK: DW_TAG_subprogram [10]
; CHECK-NEXT: DW_AT_linkage_name [DW_FORM_strx1]    (indexed (0000000b) string = "_ZNO7deleted4funcEv")
; CHECK-NEXT: DW_AT_name [DW_FORM_strx1]    (indexed (0000000c) string = "func")
; CHECK:  DW_AT_deleted [DW_FORM_flag_present]  (true)

; CHECK: DW_TAG_subprogram [11]
; CHECK-NEXT: DW_AT_linkage_name [DW_FORM_strx1]    (indexed (0000000d) string = "_ZN7deleted3barEv")
; CHECK-NEXT: DW_AT_name [DW_FORM_strx1]    (indexed (0000000e) string = "bar")
; CHECK:  DW_AT_deleted [DW_FORM_flag_present]  (true)

; ModuleID = 'debug-info-deleted.cpp'
source_filename = "debug-info-deleted.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.deleted = type { i8 }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z3foov() #0 !dbg !7 {
  %1 = alloca %class.deleted, align 1
  call void @llvm.dbg.declare(metadata ptr %1, metadata !10, metadata !DIExpression()), !dbg !39
  ret void, !dbg !40
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone uwtable }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git 715c47d5de9aa8860050992a7aaf27dca53f7f4a)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "debug-info-deleted.cpp", directory: "/home/sourabh/work/dwarf/c_c++/c++11", checksumkind: CSK_MD5, checksum: "49dc56907586479c64634558b060292d")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git 715c47d5de9aa8860050992a7aaf27dca53f7f4a)"}
!7 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 14, type: !8, scopeLine: 14, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocalVariable(name: "obj1", scope: !7, file: !1, line: 15, type: !11)
!11 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "deleted", file: !1, line: 1, size: 8, flags: DIFlagTypePassByReference, elements: !12, identifier: "_ZTS7deleted")
!12 = !{!13, !17, !22, !26, !30, !33, !34, !36}
!13 = !DISubprogram(name: "deleted", scope: !11, file: !1, line: 3, type: !14, scopeLine: 3, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!14 = !DISubroutineType(types: !15)
!15 = !{null, !16}
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!17 = !DISubprogram(name: "deleted", scope: !11, file: !1, line: 5, type: !18, scopeLine: 5, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDeleted)
!18 = !DISubroutineType(types: !19)
!19 = !{null, !16, !20}
!20 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !21, size: 64)
!21 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !11)
!22 = !DISubprogram(name: "operator=", linkageName: "_ZN7deletedaSERKS_", scope: !11, file: !1, line: 6, type: !23, scopeLine: 6, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDeleted)
!23 = !DISubroutineType(types: !24)
!24 = !{!25, !16, !20}
!25 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !11, size: 64)
!26 = !DISubprogram(name: "deleted", scope: !11, file: !1, line: 8, type: !27, scopeLine: 8, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDeleted)
!27 = !DISubroutineType(types: !28)
!28 = !{null, !16, !29}
!29 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: !11, size: 64)
!30 = !DISubprogram(name: "operator=", linkageName: "_ZN7deletedaSEOS_", scope: !11, file: !1, line: 9, type: !31, scopeLine: 9, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDeleted)
!31 = !DISubroutineType(types: !32)
!32 = !{!25, !16, !29}
!33 = !DISubprogram(name: "~deleted", scope: !11, file: !1, line: 11, type: !14, scopeLine: 11, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!34 = !DISubprogram(name: "func", linkageName: "_ZNO7deleted4funcEv", scope: !11, file: !1, line: 13, type: !35, scopeLine: 13, flags: DIFlagPublic | DIFlagPrototyped | DIFlagRValueReference, spFlags: DISPFlagDeleted)                                                                                                                                                                                                           
!35 = !DISubroutineType(flags: DIFlagRValueReference, types: !15)                                                                                                                                                 
!36 = !DISubprogram(name: "bar", linkageName: "_ZN7deleted3barEv", scope: !11, file: !1, line: 15, type: !37, scopeLine: 15, flags: DIFlagPublic | DIFlagPrototyped | DIFlagStaticMember, spFlags: DISPFlagDeleted)                                                                                                                                                                                                                 
!37 = !DISubroutineType(types: !38)
!38 = !{null}
!39 = !DILocation(line: 15, column: 13, scope: !7)
!40 = !DILocation(line: 16, column: 3, scope: !7)
