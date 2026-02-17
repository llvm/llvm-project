; RUN: opt %loadNPMPolly '-passes=polly-custom<detect>' -pass-remarks-missed=polly-detect -disable-output < %s 2>&1 | FileCheck %s

; CHECK: remark: /app/example.c:73:19: The following errors keep this region from being a Scop.
; CHECK: remark: /app/example.c:73:19: Incompatible (non-fixed size) type: <vscale x 8 x half>
; CHECK: remark: /app/example.c:73:19: Invalid Scop candidate ends here.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64"

define void @sve(i32 %n, ptr noalias nonnull %A) {
entry:
  br label %for

  for:
    %j = phi i32 [ 0, %entry ], [ %j.inc, %inc ]
    %j.cmp = icmp slt i32 %j, %n
    br i1 %j.cmp, label %body, label %exit

    body:
      %ld = load <vscale x 8 x half>, ptr %A, align 16, !dbg !38, !tbaa !53
        #dbg_value(<vscale x 8 x half> %ld, !46, !DIExpression(), !55)
      br label %inc

  inc:
    %j.inc = add nuw nsw i32 %j, 1
    br label %for

  exit:
    br label %return

return:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!27, !28, !29, !30, !31, !32}
!llvm.ident = !{!33}
!llvm.errno.tbaa = !{!34}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 23.0.0git (/home/meinersbur/src/llvm/main/_src/clang dcb6e15a83c907be0e71e01f3509d530135c56ae)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, globals: !9, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "../issue177859.c", directory: "/home/meinersbur/src/llvm/main/release", checksumkind: CSK_MD5, checksum: "cabf91caf1d9144496f767cef582f50d")
!2 = !{!3, !6}
!3 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64)
!4 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !5)
!5 = !DIBasicType(name: "__fp16", size: 16, encoding: DW_ATE_float)
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "ggml_float", file: !7, line: 48, baseType: !8)
!7 = !DIFile(filename: "/app/example.c", directory: "")
!8 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!9 = !{!10, !16, !21}
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(scope: null, file: !7, line: 51, type: !12, isLocal: true, isDefinition: true)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 72, elements: !14)
!13 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_unsigned_char)
!14 = !{!15}
!15 = !DISubrange(count: 9)
!16 = !DIGlobalVariableExpression(var: !17, expr: !DIExpression())
!17 = distinct !DIGlobalVariable(scope: null, file: !7, line: 51, type: !18, isLocal: true, isDefinition: true)
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 120, elements: !19)
!19 = !{!20}
!20 = !DISubrange(count: 15)
!21 = !DIGlobalVariableExpression(var: !22, expr: !DIExpression())
!22 = distinct !DIGlobalVariable(scope: null, file: !7, line: 51, type: !23, isLocal: true, isDefinition: true)
!23 = !DICompositeType(tag: DW_TAG_array_type, baseType: !24, size: 952, elements: !25)
!24 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !13)
!25 = !{!26}
!26 = !DISubrange(count: 119)
!27 = !{i32 7, !"Dwarf Version", i32 5}
!28 = !{i32 2, !"Debug Info Version", i32 3}
!29 = !{i32 1, !"wchar_size", i32 4}
!30 = !{i32 7, !"uwtable", i32 2}
!31 = !{i32 7, !"frame-pointer", i32 4}
!32 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!33 = !{!"clang version 23.0.0git (/home/meinersbur/src/llvm/main/_src/clang dcb6e15a83c907be0e71e01f3509d530135c56ae)"}
!34 = !{!35, !35, i64 0}
!35 = !{!"int", !36, i64 0}
!36 = !{!"omnipotent char", !37, i64 0}
!37 = !{!"Simple C/C++ TBAA"}
!38 = !DILocation(line: 73, column: 19, scope: !39, atomGroup: 74, atomRank: 2)
!39 = distinct !DILexicalBlock(scope: !40, file: !7, line: 71, column: 53)
!40 = distinct !DILexicalBlock(scope: !41, file: !7, line: 71, column: 9)
!41 = distinct !DILexicalBlock(scope: !42, file: !7, line: 71, column: 9)
!42 = distinct !DISubprogram(name: "ggml_vec_dot_f16", scope: !7, file: !7, line: 50, type: !43, scopeLine: 50, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !45, keyInstructions: true)
!43 = !DISubroutineType(types: !44)
!44 = !{null}
!45 = !{!46}
!46 = !DILocalVariable(name: "ay1", scope: !42, file: !7, line: 70, type: !47)
!47 = !DIDerivedType(tag: DW_TAG_typedef, name: "svfloat16_t", file: !48, line: 36, baseType: !49)
!48 = !DIFile(filename: "/cefs/9b/9b63fa585dbaa73402339c99_clang-trunk-20260125/lib/clang/23/include/arm_sve.h", directory: "")
!49 = !DIDerivedType(tag: DW_TAG_typedef, name: "__SVFloat16_t", file: !1, baseType: !50)
!50 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, flags: DIFlagVector, elements: !51)
!51 = !{!52}
!52 = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_constu, 4, DW_OP_bregx, 46, 0, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
!53 = !{!54, !54, i64 0}
!54 = !{!"__fp16", !36, i64 0}
!55 = !DILocation(line: 0, scope: !42)
