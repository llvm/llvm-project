; RUN: llc < %s -verify-machineinstrs
; PR16110
;
; This test case contains a value that is split into two connected components
; by rematerialization during coalescing. It also contains a DBG_VALUE
; instruction which must be updated during
; ConnectedVNInfoEqClasses::Distribute().

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios3.0.0"

@c = common global i32 0, align 4
@b = common global i32 0, align 4
@a = common global i64 0, align 8
@d = common global i32 0, align 4

; Function Attrs: nounwind ssp
define i32 @pr16110() #0 !dbg !4 {
for.cond1.preheader:
  store i32 0, i32* @c, align 4, !dbg !21
  br label %for.cond1.outer, !dbg !26

for.cond1:                                        ; preds = %for.end9, %for.cond1.outer
  %storemerge11 = phi i32 [ 0, %for.end9 ], [ %storemerge11.ph, %for.cond1.outer ]
  %cmp = icmp slt i32 %storemerge11, 1, !dbg !26
  br i1 %cmp, label %for.body2, label %for.end9, !dbg !26

for.body2:                                        ; preds = %for.cond1
  store i32 %storemerge11, i32* @b, align 4, !dbg !26
  tail call void @llvm.dbg.value(metadata i32* null, i64 0, metadata !11, metadata !DIExpression()), !dbg !28
  %0 = load i64, i64* @a, align 8, !dbg !29
  %xor = xor i64 %0, %e.1.ph, !dbg !29
  %conv3 = trunc i64 %xor to i32, !dbg !29
  tail call void @llvm.dbg.value(metadata i32 %conv3, i64 0, metadata !10, metadata !DIExpression()), !dbg !29
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
  %3 = load i32, i32* @d, align 4, !dbg !31
  %tobool10 = icmp eq i32 %3, 0, !dbg !31
  br label %for.cond1

for.end9:                                         ; preds = %for.cond1
  br i1 %tobool10, label %if.end, label %for.cond1, !dbg !31

if.end:                                           ; preds = %for.end9
  store i32 %storemerge11, i32* @b, align 4, !dbg !26
  ret i32 0, !dbg !32
}

declare i32 @fn2(...) #1

declare i32 @fn3(...) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind ssp "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!33}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.4 (trunk 182024) (llvm/trunk 182023)", isOptimized: true, emissionKind: 0, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !15, imports: !2)
!1 = !DIFile(filename: "pr16110.c", directory: "/d/b")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "pr16110", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, scopeLine: 7, file: !1, scope: !5, type: !6, variables: !9)
!5 = !DIFile(filename: "pr16110.c", directory: "/d/b")
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{!10, !11}
!10 = !DILocalVariable(name: "e", line: 8, scope: !4, file: !5, type: !8)
!11 = !DILocalVariable(name: "f", line: 13, scope: !12, file: !5, type: !14)
!12 = distinct !DILexicalBlock(line: 12, column: 0, file: !1, scope: !13)
!13 = distinct !DILexicalBlock(line: 12, column: 0, file: !1, scope: !4)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, baseType: !8)
!15 = !{!16, !18, !19, !20}
!16 = !DIGlobalVariable(name: "a", line: 1, isLocal: false, isDefinition: true, scope: null, file: !5, type: !17, variable: i64* @a)
!17 = !DIBasicType(tag: DW_TAG_base_type, name: "long long int", size: 64, align: 32, encoding: DW_ATE_signed)
!18 = !DIGlobalVariable(name: "b", line: 2, isLocal: false, isDefinition: true, scope: null, file: !5, type: !8, variable: i32* @b)
!19 = !DIGlobalVariable(name: "c", line: 3, isLocal: false, isDefinition: true, scope: null, file: !5, type: !8, variable: i32* @c)
!20 = !DIGlobalVariable(name: "d", line: 4, isLocal: false, isDefinition: true, scope: null, file: !5, type: !8, variable: i32* @d)
!21 = !DILocation(line: 10, scope: !22)
!22 = distinct !DILexicalBlock(line: 10, column: 0, file: !1, scope: !4)
!26 = !DILocation(line: 12, scope: !13)
!27 = !{i32* null}
!28 = !DILocation(line: 13, scope: !12)
!29 = !DILocation(line: 14, scope: !12)
!31 = !DILocation(line: 16, scope: !4)
!32 = !DILocation(line: 18, scope: !4)
!33 = !{i32 1, !"Debug Info Version", i32 3}
