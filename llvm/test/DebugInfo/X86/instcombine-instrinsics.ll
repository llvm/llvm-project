; RUN: opt %s -O2 -S -o - | FileCheck %s
; Verify that we emit the same intrinsic at most once.
; rdar://problem/13056109
;
; CHECK: call void @llvm.dbg.value(metadata %struct.i14** %p
; CHECK-NOT: call void @llvm.dbg.value(metadata %struct.i14** %p
; CHECK-NEXT: call i32 @foo
; CHECK: ret
;
;
; typedef struct {
;   long i;
; } i14;
;
; int foo(i14**);
;
;   void init() {
;     i14* p = 0;
;     foo(&p);
;     p->i |= 4;
;     foo(&p);
;   }
;
; ModuleID = 'instcombine_intrinsics.c'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%struct.i14 = type { i64 }

; Function Attrs: nounwind ssp uwtable
define void @init() #0 !dbg !4 {
  %p = alloca %struct.i14*, align 8
  call void @llvm.dbg.declare(metadata %struct.i14** %p, metadata !11, metadata !DIExpression()), !dbg !18
  store %struct.i14* null, %struct.i14** %p, align 8, !dbg !18
  %1 = call i32 @foo(%struct.i14** %p), !dbg !19
  %2 = load %struct.i14*, %struct.i14** %p, align 8, !dbg !20
  %3 = getelementptr inbounds %struct.i14, %struct.i14* %2, i32 0, i32 0, !dbg !20
  %4 = load i64, i64* %3, align 8, !dbg !20
  %5 = or i64 %4, 4, !dbg !20
  store i64 %5, i64* %3, align 8, !dbg !20
  %6 = call i32 @foo(%struct.i14** %p), !dbg !21
  ret void, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i32 @foo(%struct.i14**)

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "instcombine_intrinsics.c", directory: "")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "init", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, scopeLine: 7, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "instcombine_intrinsics.c", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{i32 2, !"Dwarf Version", i32 2}
!9 = !{i32 1, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.5.0 "}
!11 = !DILocalVariable(name: "p", line: 8, scope: !4, file: !5, type: !12)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !13)
!13 = !DIDerivedType(tag: DW_TAG_typedef, name: "i14", line: 3, file: !1, baseType: !14)
!14 = !DICompositeType(tag: DW_TAG_structure_type, line: 1, size: 64, align: 64, file: !1, elements: !15)
!15 = !{!16}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "i", line: 2, size: 64, align: 64, file: !1, scope: !14, baseType: !17)
!17 = !DIBasicType(tag: DW_TAG_base_type, name: "long int", size: 64, align: 64, encoding: DW_ATE_signed)
!18 = !DILocation(line: 8, scope: !4)
!19 = !DILocation(line: 9, scope: !4)
!20 = !DILocation(line: 10, scope: !4)
!21 = !DILocation(line: 11, scope: !4)
!22 = !DILocation(line: 12, scope: !4)
