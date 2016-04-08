; RUN: llc < %s - | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-macosx10.6.7"

;CHECK: 	vadd.f32	q4, q8, q8
;CHECK-NEXT: Ltmp1
;CHECK-NEXT: LBB0_1

;CHECK:@DEBUG_VALUE: x <- %Q4{{$}}
;CHECK-NEXT:@DEBUG_VALUE: y <- %Q4{{$}}


@.str = external constant [13 x i8]

declare <4 x float> @test0001(float) nounwind readnone ssp

define i32 @main(i32 %argc, i8** nocapture %argv, i1 %cond) nounwind ssp !dbg !10 {
entry:
  br label %for.body9

for.body9:                                        ; preds = %for.body9, %entry
  %add19 = fadd <4 x float> undef, <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, !dbg !39
  tail call void @llvm.dbg.value(metadata <4 x float> %add19, i64 0, metadata !27, metadata !DIExpression()), !dbg !39
  %add20 = fadd <4 x float> undef, <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, !dbg !39
  tail call void @llvm.dbg.value(metadata <4 x float> %add20, i64 0, metadata !28, metadata !DIExpression()), !dbg !39
  br i1 %cond, label %for.end54, label %for.body9, !dbg !44

for.end54:                                        ; preds = %for.body9
  %tmp115 = extractelement <4 x float> %add19, i32 1
  %conv6.i75 = fpext float %tmp115 to double, !dbg !45
  %call.i82 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i32 0, i32 0), double undef, double %conv6.i75, double undef, double undef) nounwind, !dbg !45
  %tmp116 = extractelement <4 x float> %add20, i32 1
  %conv6.i76 = fpext float %tmp116 to double, !dbg !45
  %call.i83 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i32 0, i32 0), double undef, double %conv6.i76, double undef, double undef) nounwind, !dbg !45
  ret i32 0, !dbg !49
}

declare i32 @printf(i8* nocapture, ...) nounwind

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.module.flags = !{!56}
!llvm.dbg.cu = !{!2}

!0 = distinct !DISubprogram(name: "test0001", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, file: !54, scope: null, type: !3, variables: !51)
!1 = !DIFile(filename: "build2.c", directory: "/private/tmp")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 (trunk 129915)", isOptimized: true, emissionKind: FullDebug, file: !54, enums: !{}, retainedTypes: !{}, subprograms: !50, imports:  null)
!3 = !DISubroutineType(types: !4)
!4 = !{!5}
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "v4f32", line: 14, file: !54, scope: !2, baseType: !6)
!6 = !DICompositeType(tag: DW_TAG_array_type, size: 128, align: 128, file: !54, scope: !2, baseType: !7, elements: !8)
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!8 = !{!9}
!9 = !DISubrange(count: 4)
!10 = distinct !DISubprogram(name: "main", line: 59, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, file: !54, scope: null, type: !11, variables: !52)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = distinct !DISubprogram(name: "printFV", line: 41, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, file: !55, scope: null, type: !16, variables: !53)
!15 = !DIFile(filename: "/Volumes/Lalgate/work/llvm/projects/llvm-test/SingleSource/UnitTests/Vector/helpers.h", directory: "/private/tmp")
!16 = !DISubroutineType(types: !17)
!17 = !{null}
!18 = !DILocalVariable(name: "a", line: 3, arg: 1, scope: !0, file: !1, type: !7)
!19 = !DILocalVariable(name: "argc", line: 59, arg: 1, scope: !10, file: !1, type: !13)
!20 = !DILocalVariable(name: "argv", line: 59, arg: 2, scope: !10, file: !1, type: !21)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, scope: !2, baseType: !22)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, scope: !2, baseType: !23)
!23 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!24 = !DILocalVariable(name: "i", line: 60, scope: !25, file: !1, type: !13)
!25 = distinct !DILexicalBlock(line: 59, column: 33, file: !1, scope: !10)
!26 = !DILocalVariable(name: "j", line: 60, scope: !25, file: !1, type: !13)
!27 = !DILocalVariable(name: "x", line: 61, scope: !25, file: !1, type: !5)
!28 = !DILocalVariable(name: "y", line: 62, scope: !25, file: !1, type: !5)
!29 = !DILocalVariable(name: "z", line: 63, scope: !25, file: !1, type: !5)
!30 = !DILocalVariable(name: "F", line: 41, arg: 1, scope: !14, file: !15, type: !31)
!31 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, scope: !2, baseType: !32)
!32 = !DIDerivedType(tag: DW_TAG_typedef, name: "FV", line: 25, file: !55, scope: !2, baseType: !33)
!33 = !DICompositeType(tag: DW_TAG_union_type, line: 22, size: 128, align: 128, file: !55, scope: !2, elements: !34)
!34 = !{!35, !37}
!35 = !DIDerivedType(tag: DW_TAG_member, name: "V", line: 23, size: 128, align: 128, file: !55, scope: !15, baseType: !36)
!36 = !DIDerivedType(tag: DW_TAG_typedef, name: "v4sf", line: 3, file: !55, scope: !2, baseType: !6)
!37 = !DIDerivedType(tag: DW_TAG_member, name: "A", line: 24, size: 128, align: 32, file: !55, scope: !15, baseType: !38)
!38 = !DICompositeType(tag: DW_TAG_array_type, size: 128, align: 32, scope: !2, baseType: !7, elements: !8)
!39 = !DILocation(line: 79, column: 7, scope: !40)
!40 = distinct !DILexicalBlock(line: 75, column: 35, file: !1, scope: !41)
!41 = distinct !DILexicalBlock(line: 75, column: 5, file: !1, scope: !42)
!42 = distinct !DILexicalBlock(line: 71, column: 32, file: !1, scope: !43)
!43 = distinct !DILexicalBlock(line: 71, column: 3, file: !1, scope: !25)
!44 = !DILocation(line: 75, column: 5, scope: !42)
!45 = !DILocation(line: 42, column: 2, scope: !46, inlinedAt: !48)
!46 = distinct !DILexicalBlock(line: 42, column: 2, file: !15, scope: !47)
!47 = distinct !DILexicalBlock(line: 41, column: 28, file: !15, scope: !14)
!48 = !DILocation(line: 95, column: 3, scope: !25)
!49 = !DILocation(line: 99, column: 3, scope: !25)
!50 = !{!0, !10, !14}
!51 = !{!18}
!52 = !{!19, !20, !24, !26, !27, !28, !29}
!53 = !{!30}
!54 = !DIFile(filename: "build2.c", directory: "/private/tmp")
!55 = !DIFile(filename: "/Volumes/Lalgate/work/llvm/projects/llvm-test/SingleSource/UnitTests/Vector/helpers.h", directory: "/private/tmp")
!56 = !{i32 1, !"Debug Info Version", i32 3}
