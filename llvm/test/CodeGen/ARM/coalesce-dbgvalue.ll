; RUN: llc < %s -verify-machineinstrs
; PR16110
;
; This test case contains a value that is split into two connected components
; by rematerialization during coalescing. It also contains a DBG_VALUE
; instruction which must be updated during
; ConnectedVNInfoEqClasses::Distribute().

source_filename = "test/CodeGen/ARM/coalesce-dbgvalue.ll"
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios3.0.0"

@c = common global i32 0, align 4, !dbg !0
@b = common global i32 0, align 4, !dbg !4
@a = common global i64 0, align 8, !dbg !6
@d = common global i32 0, align 4, !dbg !9

; Function Attrs: nounwind ssp
define i32 @pr16110() #0 !dbg !15 {
for.cond1.preheader:
  store i32 0, i32* @c, align 4, !dbg !24
  br label %for.cond1.outer, !dbg !26

for.cond1:                                        ; preds = %for.end9, %for.cond1.outer
  %storemerge11 = phi i32 [ 0, %for.end9 ], [ %storemerge11.ph, %for.cond1.outer ]
  %cmp = icmp slt i32 %storemerge11, 1, !dbg !26
  br i1 %cmp, label %for.body2, label %for.end9, !dbg !26

for.body2:                                        ; preds = %for.cond1
  store i32 %storemerge11, i32* @b, align 4, !dbg !26
  tail call void @llvm.dbg.value(metadata i32* null, metadata !20, metadata !27), !dbg !28
  %0 = load i64, i64* @a, align 8, !dbg !29
  %xor = xor i64 %0, %e.1.ph, !dbg !29
  %conv3 = trunc i64 %xor to i32, !dbg !29
  tail call void @llvm.dbg.value(metadata i32 %conv3, metadata !19, metadata !27), !dbg !29
  %tobool4 = icmp eq i32 %conv3, 0, !dbg !29
  br i1 %tobool4, label %land.end, label %land.rhs, !dbg !29

land.rhs:                                         ; preds = %for.body2
  %call = tail call i32 bitcast (i32 (...)* @fn3 to i32 ()*)() #3, !dbg !29
  %tobool5 = icmp ne i32 %call, 0, !dbg !29
  br label %land.end

land.end:                                         ; preds = %land.rhs, %for.body2
  %1 = phi i1 [ false, %for.body2 ], [ %tobool5, %land.rhs ]
  %land.ext = zext i1 %1 to i32
  %call6 = tail call i32 bitcast (i32 (...)* @fn2 to i32 (i32, i32*)*)(i32 %land.ext, i32* null) #3
  %2 = load i32, i32* @b, align 4, !dbg !26
  %inc8 = add nsw i32 %2, 1, !dbg !26
  %phitmp = and i64 %xor, 4294967295, !dbg !26
  br label %for.cond1.outer, !dbg !26

for.cond1.outer:                                  ; preds = %land.end, %for.cond1.preheader
  %storemerge11.ph = phi i32 [ %inc8, %land.end ], [ 0, %for.cond1.preheader ]
  %e.1.ph = phi i64 [ %phitmp, %land.end ], [ 0, %for.cond1.preheader ]
  %3 = load i32, i32* @d, align 4, !dbg !30
  %tobool10 = icmp eq i32 %3, 0, !dbg !30
  br label %for.cond1

for.end9:                                         ; preds = %for.cond1
  br i1 %tobool10, label %if.end, label %for.cond1, !dbg !30

if.end:                                           ; preds = %for.end9
  store i32 %storemerge11, i32* @b, align 4, !dbg !26
  ret i32 0, !dbg !31
}

declare i32 @fn2(...) #1

declare i32 @fn3(...) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind ssp "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!11}
!llvm.module.flags = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "c", scope: null, file: !2, line: 3, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "pr16110.c", directory: "/d/b")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = !DIGlobalVariable(name: "b", scope: null, file: !2, line: 2, type: !3, isLocal: false, isDefinition: true)
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = !DIGlobalVariable(name: "a", scope: null, file: !2, line: 1, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "long long int", size: 64, align: 32, encoding: DW_ATE_signed)
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = !DIGlobalVariable(name: "d", scope: null, file: !2, line: 4, type: !3, isLocal: false, isDefinition: true)
!11 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 3.4 (trunk 182024) (llvm/trunk 182023)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !12, retainedTypes: !12, globals: !13, imports: !12)
!12 = !{}
!13 = !{!6, !4, !0, !9}
!14 = !{i32 1, !"Debug Info Version", i32 3}
!15 = distinct !DISubprogram(name: "pr16110", scope: !2, file: !2, line: 7, type: !16, isLocal: false, isDefinition: true, scopeLine: 7, virtualIndex: 6, isOptimized: true, unit: !11, retainedNodes: !18)
!16 = !DISubroutineType(types: !17)
!17 = !{!3}
!18 = !{!19, !20}
!19 = !DILocalVariable(name: "e", scope: !15, file: !2, line: 8, type: !3)
!20 = !DILocalVariable(name: "f", scope: !21, file: !2, line: 13, type: !23)
!21 = distinct !DILexicalBlock(scope: !22, file: !2, line: 12)
!22 = distinct !DILexicalBlock(scope: !15, file: !2, line: 12)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 32, align: 32)
!24 = !DILocation(line: 10, scope: !25)
!25 = distinct !DILexicalBlock(scope: !15, file: !2, line: 10)
!26 = !DILocation(line: 12, scope: !22)
!27 = !DIExpression()
!28 = !DILocation(line: 13, scope: !21)
!29 = !DILocation(line: 14, scope: !21)
!30 = !DILocation(line: 16, scope: !15)
!31 = !DILocation(line: 18, scope: !15)

