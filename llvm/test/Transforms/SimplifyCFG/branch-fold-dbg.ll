; RUN: opt -simplifycfg -S < %s | FileCheck %s

%0 = type { i32*, i32* }

@0 = external hidden constant [5 x %0], align 4

define void @foo(i32) nounwind ssp !dbg !0 {
Entry:
  %1 = icmp slt i32 %0, 0, !dbg !5
  br i1 %1, label %BB5, label %BB1, !dbg !5

BB1:                                              ; preds = %Entry
  %2 = icmp sgt i32 %0, 4, !dbg !5
  br i1 %2, label %BB5, label %BB2, !dbg !5

BB2:                                              ; preds = %BB1
  %3 = shl i32 1, %0, !dbg !5
  %4 = and i32 %3, 31, !dbg !5
  %5 = icmp eq i32 %4, 0, !dbg !5
  br i1 %5, label %BB5, label %BB3, !dbg !5

;CHECK: icmp eq
;CHECK-NEXT: getelementptr
;CHECK-NEXT: icmp eq

BB3:                                              ; preds = %BB2
  %6 = getelementptr inbounds [5 x %0], [5 x %0]* @0, i32 0, i32 %0, !dbg !6
  call void @llvm.dbg.value(metadata %0* %6, i64 0, metadata !7, metadata !{}), !dbg !12
  %7 = icmp eq %0* %6, null, !dbg !13
  br i1 %7, label %BB5, label %BB4, !dbg !13

BB4:                                              ; preds = %BB3
  %8 = icmp slt i32 %0, 0, !dbg !5
  ret void, !dbg !14

BB5:                                              ; preds = %BB3, %BB2, %BB1, %Entry
  ret void, !dbg !14
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}

!0 = distinct !DISubprogram(name: "foo", line: 231, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !2, file: !15, scope: !1, type: !3)
!1 = !DIFile(filename: "a.c", directory: "/private/tmp")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang (trunk 129006)", isOptimized: true, emissionKind: FullDebug, file: !15, enums: !4, retainedTypes: !4)
!3 = !DISubroutineType(types: !4)
!4 = !{null}
!5 = !DILocation(line: 131, column: 2, scope: !0)
!6 = !DILocation(line: 134, column: 2, scope: !0)
!7 = !DILocalVariable(name: "bar", line: 232, scope: !8, file: !1, type: !9)
!8 = distinct !DILexicalBlock(line: 231, column: 1, file: !15, scope: !0)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, scope: !2, baseType: !10)
!10 = !DIDerivedType(tag: DW_TAG_const_type, scope: !2, baseType: !11)
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!12 = !DILocation(line: 232, column: 40, scope: !8)
!13 = !DILocation(line: 234, column: 2, scope: !8)
!14 = !DILocation(line: 274, column: 1, scope: !8)
!15 = !DIFile(filename: "a.c", directory: "/private/tmp")
