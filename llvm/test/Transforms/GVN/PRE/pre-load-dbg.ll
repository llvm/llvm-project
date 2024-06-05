; RUN: opt < %s -passes=gvn -gvn-max-num-insns=22 -S | FileCheck %s

; Debug information should not impact gvn. The following two functions have same
; code except debug information. They should generate same optimized
; instructions.

%struct.a = type { i16 }

@f = local_unnamed_addr global i16 0, align 1
@m = local_unnamed_addr global ptr null, align 1
@h = global %struct.a zeroinitializer, align 1

define void @withdbg() {
; CHECK-LABEL: @withdbg
; CHECK:         [[PRE_PRE1:%.*]] = load i16, ptr @f, align 1
; CHECK-NEXT:    [[PRE_PRE2:%.*]] = load ptr, ptr @m, align 1
; CHECK-NEXT:    br i1 true, label %[[BLOCK1:.*]], label %[[BLOCK2:.*]]
; CHECK:       [[BLOCK1]]:
; CHECK-NEXT:    [[CONV:%.*]] = sext i16 [[PRE_PRE1]] to i32
; CHECK-NEXT:    store i32 [[CONV]], ptr [[PRE_PRE2]], align 1

entry:
  %agg.tmp.ensured.sroa.0.i = alloca i16, align 1
  %cmp = icmp ne ptr @withdbg, null
  br i1 %cmp, label %lor.end, label %lor.rhs

lor.rhs:                                          ; preds = %entry
  call void @llvm.dbg.declare(metadata ptr undef, metadata !46, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.declare(metadata ptr undef, metadata !47, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.declare(metadata ptr undef, metadata !48, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.declare(metadata ptr undef, metadata !49, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.declare(metadata ptr undef, metadata !50, metadata !DIExpression()), !dbg !40
  %agg.tmp.ensured.sroa.0.0.copyload.i = load volatile i16, ptr @h, align 1
  store i16 %agg.tmp.ensured.sroa.0.0.copyload.i, ptr %agg.tmp.ensured.sroa.0.i, align 1
  %agg.tmp.ensured.sroa.0.0.copyload.1.i = load volatile i16, ptr @h, align 1
  store i16 %agg.tmp.ensured.sroa.0.0.copyload.1.i, ptr %agg.tmp.ensured.sroa.0.i, align 1
  %agg.tmp.ensured.sroa.0.0.copyload.2.i = load volatile i16, ptr @h, align 1
  store i16 %agg.tmp.ensured.sroa.0.0.copyload.2.i, ptr %agg.tmp.ensured.sroa.0.i, align 1
  %agg.tmp.ensured.sroa.0.0.copyload.3.i = load volatile i16, ptr @h, align 1
  store i16 %agg.tmp.ensured.sroa.0.0.copyload.3.i, ptr %agg.tmp.ensured.sroa.0.i, align 1
  %agg.tmp.ensured.sroa.0.0.copyload.4.i = load volatile i16, ptr @h, align 1
  store i16 %agg.tmp.ensured.sroa.0.0.copyload.4.i, ptr %agg.tmp.ensured.sroa.0.i, align 1
  %agg.tmp.ensured.sroa.0.0.copyload.5.i = load volatile i16, ptr @h, align 1
  store i16 %agg.tmp.ensured.sroa.0.0.copyload.5.i, ptr %agg.tmp.ensured.sroa.0.i, align 1
  %agg.tmp.ensured.sroa.0.0.copyload.6.i = load volatile i16, ptr @h, align 1
  store i16 %agg.tmp.ensured.sroa.0.0.copyload.6.i, ptr %agg.tmp.ensured.sroa.0.i, align 1
  %agg.tmp.ensured.sroa.0.0.copyload.7.i = load volatile i16, ptr @h, align 1
  store i16 %agg.tmp.ensured.sroa.0.0.copyload.7.i, ptr %agg.tmp.ensured.sroa.0.i, align 1
  %agg.tmp.ensured.sroa.0.0.copyload.8.i = load volatile i16, ptr @h, align 1
  store i16 %agg.tmp.ensured.sroa.0.0.copyload.8.i, ptr %agg.tmp.ensured.sroa.0.i, align 1
  %fvalue = load i16, ptr @f, align 1
  %mvalue = load ptr, ptr @m, align 1
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %entry
  %tmp11 = load i16, ptr @f, align 1
  %conv.i.i6 = sext i16 %tmp11 to i32
  %tmp12 = load ptr, ptr @m, align 1
  store i32 %conv.i.i6, ptr %tmp12, align 1
  ret void
}

define void @lessdbg() {
; CHECK-LABEL: @lessdbg
; CHECK:         [[PRE_PRE1:%.*]] = load i16, ptr @f, align 1
; CHECK-NEXT:    [[PRE_PRE2:%.*]] = load ptr, ptr @m, align 1
; CHECK-NEXT:    br i1 true, label %[[BLOCK1:.*]], label %[[BLOCK2:.*]]
; CHECK:       [[BLOCK1]]:
; CHECK-NEXT:    [[CONV:%.*]] = sext i16 [[PRE_PRE1]] to i32
; CHECK-NEXT:    store i32 [[CONV]], ptr [[PRE_PRE2]], align 1

entry:
  %agg.tmp.ensured.sroa.0.i = alloca i16, align 1
  %cmp = icmp ne ptr @lessdbg, null
  br i1 %cmp, label %lor.end, label %lor.rhs

lor.rhs:                                          ; preds = %entry
  %agg.tmp.ensured.sroa.0.0.copyload.i = load volatile i16, ptr @h, align 1
  store i16 %agg.tmp.ensured.sroa.0.0.copyload.i, ptr %agg.tmp.ensured.sroa.0.i, align 1
  %agg.tmp.ensured.sroa.0.0.copyload.1.i = load volatile i16, ptr @h, align 1
  store i16 %agg.tmp.ensured.sroa.0.0.copyload.1.i, ptr %agg.tmp.ensured.sroa.0.i, align 1
  %agg.tmp.ensured.sroa.0.0.copyload.2.i = load volatile i16, ptr @h, align 1
  store i16 %agg.tmp.ensured.sroa.0.0.copyload.2.i, ptr %agg.tmp.ensured.sroa.0.i, align 1
  %agg.tmp.ensured.sroa.0.0.copyload.3.i = load volatile i16, ptr @h, align 1
  store i16 %agg.tmp.ensured.sroa.0.0.copyload.3.i, ptr %agg.tmp.ensured.sroa.0.i, align 1
  %agg.tmp.ensured.sroa.0.0.copyload.4.i = load volatile i16, ptr @h, align 1
  store i16 %agg.tmp.ensured.sroa.0.0.copyload.4.i, ptr %agg.tmp.ensured.sroa.0.i, align 1
  %agg.tmp.ensured.sroa.0.0.copyload.5.i = load volatile i16, ptr @h, align 1
  store i16 %agg.tmp.ensured.sroa.0.0.copyload.5.i, ptr %agg.tmp.ensured.sroa.0.i, align 1
  %agg.tmp.ensured.sroa.0.0.copyload.6.i = load volatile i16, ptr @h, align 1
  store i16 %agg.tmp.ensured.sroa.0.0.copyload.6.i, ptr %agg.tmp.ensured.sroa.0.i, align 1
  %agg.tmp.ensured.sroa.0.0.copyload.7.i = load volatile i16, ptr @h, align 1
  store i16 %agg.tmp.ensured.sroa.0.0.copyload.7.i, ptr %agg.tmp.ensured.sroa.0.i, align 1
  %agg.tmp.ensured.sroa.0.0.copyload.8.i = load volatile i16, ptr @h, align 1
  store i16 %agg.tmp.ensured.sroa.0.0.copyload.8.i, ptr %agg.tmp.ensured.sroa.0.i, align 1
  %fvalue = load i16, ptr @f, align 1
  %mvalue = load ptr, ptr @m, align 1
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %entry
  %tmp11 = load i16, ptr @f, align 1
  %conv.i.i6 = sext i16 %tmp11 to i32
  %tmp12 = load ptr, ptr @m, align 1
  store i32 %conv.i.i6, ptr %tmp12, align 1
  ret void
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!35, !36}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 17.0.0.prerel", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "bbi-78272.c", directory: "/tmp")
!5 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)

!35 = !{i32 7, !"Dwarf Version", i32 4}
!36 = !{i32 2, !"Debug Info Version", i32 3}
!40 = !DILocation(line: 15, column: 7, scope: !41)
!41 = distinct !DISubprogram(name: "x", scope: !1, file: !1, line: 14, type: !42, scopeLine: 14, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !45)
!42 = !DISubroutineType(types: !43)
!43 = !{!5}
!45 = !{!46, !47, !48, !49, !50}
!46 = !DILocalVariable(name: "t", scope: !41, file: !1, line: 15, type: !5)
!47 = !DILocalVariable(name: "c", scope: !41, file: !1, line: 15, type: !5)
!48 = !DILocalVariable(name: "v", scope: !41, file: !1, line: 15, type: !5)
!49 = !DILocalVariable(name: "d", scope: !41, file: !1, line: 15, type: !5)
!50 = !DILocalVariable(name: "u", scope: !41, file: !1, line: 16, type: !5)
