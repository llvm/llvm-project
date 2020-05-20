; RUN: llc %s -o - | FileCheck %s

; This test case caused UDTs to be discovered during UDT emission, which can
; cause iterator invalidation.

; Based on this C++:
; typedef int a;
; struct b;
; class c {
;   c();
;   a b::*d;
; };
; c::c() = default;

; Previously there was an issue were the "a" typedef would be emitted twice.
; Check that there are only two typedefs, a and c.
; CHECK:        .short  4360        # Record kind: S_UDT
; CHECK:        .long   {{.*}}      # Type
; CHECK:        .asciz  "a"
; CHECK:        .p2align        2
; CHECK:        .short  4360        # Record kind: S_UDT
; CHECK:        .long   {{.*}}      # Type
; CHECK:        .asciz  "c"
; CHECK:        .p2align        2
;   No other S_UDTs.
; CHECK-NOT: S_UDT
; CHECK:        .cv_filechecksums               


; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.16.0"

%class.c = type { { i32, i32, i32 } }

; Function Attrs: noinline nounwind optnone
define dso_local %class.c* @"??0c@@AEAA@XZ"(%class.c* returned %this) unnamed_addr #0 align 2 !dbg !7 {
entry:
  %this.addr = alloca %class.c*, align 8
  store %class.c* %this, %class.c** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %class.c** %this.addr, metadata !20, metadata !DIExpression()), !dbg !22
  %this1 = load %class.c*, %class.c** %this.addr, align 8
  ret %class.c* %this1, !dbg !23
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 11.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "C:/src/llvm-project/build", checksumkind: CSK_MD5, checksum: "4cbca1b19718cc292886f5df0b72cf37")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{!"clang version 11.0.0 "}
!7 = distinct !DISubprogram(name: "c", linkageName: "??0c@@AEAA@XZ", scope: !9, file: !8, line: 7, type: !17, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !16, retainedNodes: !2)
!8 = !DIFile(filename: "t.cpp", directory: "C:/src/llvm-project/build", checksumkind: CSK_MD5, checksum: "4cbca1b19718cc292886f5df0b72cf37")
!9 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "c", file: !8, line: 3, size: 96, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !10, identifier: ".?AVc@@")
!10 = !{!11, !16}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !9, file: !8, line: 5, baseType: !12, size: 96)
!12 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !13, size: 96, extraData: !15)
!13 = !DIDerivedType(tag: DW_TAG_typedef, name: "a", file: !8, line: 1, baseType: !14)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DICompositeType(tag: DW_TAG_structure_type, name: "b", file: !8, line: 2, flags: DIFlagFwdDecl | DIFlagNonTrivial, identifier: ".?AUb@@")
!16 = !DISubprogram(name: "c", scope: !9, file: !8, line: 4, type: !17, scopeLine: 4, flags: DIFlagPrototyped, spFlags: 0)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!20 = !DILocalVariable(name: "this", arg: 1, scope: !7, type: !21, flags: DIFlagArtificial | DIFlagObjectPointer)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!22 = !DILocation(line: 0, scope: !7)
!23 = !DILocation(line: 7, scope: !7)
