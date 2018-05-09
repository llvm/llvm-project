; RUN: llc -mtriple=x86_64-apple-darwin %s -o - -filetype=obj | llvm-dwarfdump - | FileCheck %s
; Test debug info for multidimensional arrays.
;
; void f(int i, int j, int k, int r) {
;  int tensor1[i][j][k][r];
;  int tensor2[i][j][k][r];
;  use(tensor1, tensor2);
;}
;
; CHECK:        DW_TAG_array_type
; CHECK-NEXT:     DW_AT_type	(0x000000f8 "int")
; CHECK-NOT: TAG
; CHECK:          DW_TAG_subrange_type
; CHECK-NEXT:       DW_AT_type	(0x{{.*}}"__ARRAY_SIZE_TYPE__")
; CHECK-NEXT:       DW_AT_count	(0x{{.*}})
; CHECK-NOT: TAG
; CHECK:          DW_TAG_subrange_type
; CHECK-NEXT:       DW_AT_type	(0x{{.*}} "__ARRAY_SIZE_TYPE__")
; CHECK-NEXT:       DW_AT_count	(0x{{.*}})
; CHECK-NOT: TAG
; CHECK:          DW_TAG_subrange_type
; CHECK-NEXT:       DW_AT_type	(0x{{.*}} "__ARRAY_SIZE_TYPE__")
; CHECK-NEXT:       DW_AT_count	(0x{{.*}})
; CHECK-NOT: TAG
; CHECK:          DW_TAG_subrange_type
; CHECK-NEXT:       DW_AT_type	(0x{{.*}} "__ARRAY_SIZE_TYPE__")
; CHECK-NEXT:       DW_AT_count	(0x{{.*}})
; CHECK:        DW_TAG_array_type
; CHECK-NEXT:     DW_AT_type	(0x000000f8 "int")
; CHECK-NOT: TAG
; CHECK:          DW_TAG_subrange_type
; CHECK-NEXT:       DW_AT_type	(0x{{.*}}"__ARRAY_SIZE_TYPE__")
; CHECK-NEXT:       DW_AT_count	(0x{{.*}})
; CHECK-NOT: TAG
; CHECK:          DW_TAG_subrange_type
; CHECK-NEXT:       DW_AT_type	(0x{{.*}} "__ARRAY_SIZE_TYPE__")
; CHECK-NEXT:       DW_AT_count	(0x{{.*}})
; CHECK-NOT: TAG
; CHECK:          DW_TAG_subrange_type
; CHECK-NEXT:       DW_AT_type	(0x{{.*}} "__ARRAY_SIZE_TYPE__")
; CHECK-NEXT:       DW_AT_count	(0x{{.*}})
; CHECK-NOT: TAG
; CHECK:          DW_TAG_subrange_type
; CHECK-NEXT:       DW_AT_type	(0x{{.*}} "__ARRAY_SIZE_TYPE__")
; CHECK-NEXT:       DW_AT_count	(0x{{.*}})


source_filename = "/tmp/test.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

define void @f(i32 %i, i32 %j, i32 %k, i32 %r) !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata i32 %i, metadata !39, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 %j, metadata !38, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 %k, metadata !37, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 %r, metadata !36, metadata !DIExpression()), !dbg !40
  %0 = zext i32 %i to i64, !dbg !40
  %1 = zext i32 %j to i64, !dbg !40
  %2 = zext i32 %k to i64, !dbg !40
  %3 = zext i32 %r to i64, !dbg !40
  %4 = mul nuw i64 %1, %0, !dbg !40
  %5 = mul nuw i64 %4, %2, !dbg !40
  %6 = mul nuw i64 %5, %3, !dbg !40
  %vla = alloca i32, i64 %6, align 16, !dbg !40
  call void @llvm.dbg.declare(metadata i32* %vla, metadata !25, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.declare(metadata i32* %vla4, metadata !13, metadata !DIExpression()), !dbg !40
  %vla4 = alloca i32, i64 %6, align 16, !dbg !40
  call void @llvm.dbg.value(metadata i32 %i, metadata !29, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 %j, metadata !31, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 %k, metadata !33, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 %r, metadata !35, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 %i, metadata !17, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 %j, metadata !20, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i32 %k, metadata !22, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.value(metadata i64 %3, metadata !24, metadata !DIExpression()), !dbg !40
  %call = call i32 (i32*, i32*, ...) bitcast (i32 (...)* @use to i32 (i32*, i32*, ...)*)(i32* nonnull %vla, i32* nonnull %vla4) #1, !dbg !40
  ret void, !dbg !40
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

declare i32 @use(...) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { minsize nounwind optsize }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0 (trunk 324259) (llvm/trunk 324261)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/test.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 7.0.0 (trunk 324259) (llvm/trunk 324261)"}
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11, !11, !11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13, !24, !22, !20, !17, !25, !35, !33, !31, !29, !36, !37, !38, !39}
!13 = !DILocalVariable(name: "tensor2", scope: !8, file: !1, line: 3, type: !14)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, elements: !15)
!15 = !{!16, !19, !21, !23}
!16 = !DISubrange(count: !17)
!17 = !DILocalVariable(name: "vla_expr5", scope: !8, file: !1, line: 3, type: !18)
!18 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!19 = !DISubrange(count: !20)
!20 = !DILocalVariable(name: "vla_expr6", scope: !8, file: !1, line: 3, type: !18)
!21 = !DISubrange(count: !22)
!22 = !DILocalVariable(name: "vla_expr7", scope: !8, file: !1, line: 3, type: !18)
!23 = !DISubrange(count: !24)
!24 = !DILocalVariable(name: "vla_expr8", scope: !8, file: !1, line: 3, type: !18)
!25 = !DILocalVariable(name: "tensor1", scope: !8, file: !1, line: 2, type: !26)
!26 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, elements: !27)
!27 = !{!28, !30, !32, !34}
!28 = !DISubrange(count: !29)
!29 = !DILocalVariable(name: "vla_expr", scope: !8, file: !1, line: 2, type: !18)
!30 = !DISubrange(count: !31)
!31 = !DILocalVariable(name: "vla_expr1", scope: !8, file: !1, line: 2, type: !18)
!32 = !DISubrange(count: !33)
!33 = !DILocalVariable(name: "vla_expr2", scope: !8, file: !1, line: 2, type: !18)
!34 = !DISubrange(count: !35)
!35 = !DILocalVariable(name: "vla_expr3", scope: !8, file: !1, line: 2, type: !18)
!36 = !DILocalVariable(name: "r", arg: 4, scope: !8, file: !1, line: 1, type: !11)
!37 = !DILocalVariable(name: "k", arg: 3, scope: !8, file: !1, line: 1, type: !11)
!38 = !DILocalVariable(name: "j", arg: 2, scope: !8, file: !1, line: 1, type: !11)
!39 = !DILocalVariable(name: "i", arg: 1, scope: !8, file: !1, line: 1, type: !11)
!40 = !DILocation(line: 2, column: 3, scope: !8)
