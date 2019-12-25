;RUN: opt < %s -loop-reroll -S | FileCheck %s
;void foo(float * restrict a, float * restrict b, int n) {
;  for(int i = 0; i < n; i+=4) {
;    a[i] = b[i];
;    a[i+1] = b[i+1];
;    a[i+2] = b[i+2];
;    a[i+3] = b[i+3];
;  }
;}
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv4t--linux-gnueabi"

; Function Attrs: nounwind
define void @foo(float* noalias nocapture %a, float* noalias nocapture readonly %b, i32 %n) #0 !dbg !4 {
entry:
;CHECK-LABEL: @foo

  tail call void @llvm.dbg.value(metadata float* %a, metadata !12, metadata !22), !dbg !23
  tail call void @llvm.dbg.value(metadata float* %b, metadata !13, metadata !22), !dbg !24
  tail call void @llvm.dbg.value(metadata i32 %n, metadata !14, metadata !22), !dbg !25
  tail call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !22), !dbg !26
  %cmp.30 = icmp sgt i32 %n, 0, !dbg !27
  br i1 %cmp.30, label %for.body.preheader, label %for.cond.cleanup, !dbg !29

for.body.preheader:                               ; preds = %entry
  br label %for.body, !dbg !30

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup, !dbg !32

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void, !dbg !32

for.body:                                         ; preds = %for.body.preheader, %for.body
;CHECK: for.body:
;CHECK: %indvar = phi i32 [ %indvar.next, %for.body ], [ 0, {{.*}} ]
;CHECK: load
;CHECK: store
;CHECK-NOT: load
;CHECK-NOT: store
;CHECK: call void @llvm.dbg.value
;CHECK: %indvar.next = add i32 %indvar, 1
;CHECK: icmp eq i32 %indvar
  %i.031 = phi i32 [ %add13, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds float, float* %b, i32 %i.031, !dbg !30
  %0 = bitcast float* %arrayidx to i32*, !dbg !30
  %1 = load i32, i32* %0, align 4, !dbg !30, !tbaa !33
  %arrayidx1 = getelementptr inbounds float, float* %a, i32 %i.031, !dbg !37
  %2 = bitcast float* %arrayidx1 to i32*, !dbg !38
  store i32 %1, i32* %2, align 4, !dbg !38, !tbaa !33
  %add = or i32 %i.031, 1, !dbg !39
  %arrayidx2 = getelementptr inbounds float, float* %b, i32 %add, !dbg !40
  %3 = bitcast float* %arrayidx2 to i32*, !dbg !40
  %4 = load i32, i32* %3, align 4, !dbg !40, !tbaa !33
  %arrayidx4 = getelementptr inbounds float, float* %a, i32 %add, !dbg !41
  %5 = bitcast float* %arrayidx4 to i32*, !dbg !42
  store i32 %4, i32* %5, align 4, !dbg !42, !tbaa !33
  %add5 = or i32 %i.031, 2, !dbg !43
  %arrayidx6 = getelementptr inbounds float, float* %b, i32 %add5, !dbg !44
  %6 = bitcast float* %arrayidx6 to i32*, !dbg !44
  %7 = load i32, i32* %6, align 4, !dbg !44, !tbaa !33
  %arrayidx8 = getelementptr inbounds float, float* %a, i32 %add5, !dbg !45
  %8 = bitcast float* %arrayidx8 to i32*, !dbg !46
  store i32 %7, i32* %8, align 4, !dbg !46, !tbaa !33
  %add9 = or i32 %i.031, 3, !dbg !47
  %arrayidx10 = getelementptr inbounds float, float* %b, i32 %add9, !dbg !48
  %9 = bitcast float* %arrayidx10 to i32*, !dbg !48
  %10 = load i32, i32* %9, align 4, !dbg !48, !tbaa !33
  %arrayidx12 = getelementptr inbounds float, float* %a, i32 %add9, !dbg !49
  %11 = bitcast float* %arrayidx12 to i32*, !dbg !50
  store i32 %10, i32* %11, align 4, !dbg !50, !tbaa !33
  %add13 = add nuw nsw i32 %i.031, 4, !dbg !51
  tail call void @llvm.dbg.value(metadata i32 %add13, metadata !15, metadata !22), !dbg !26
  %cmp = icmp slt i32 %add13, %n, !dbg !27
  br i1 %cmp, label %for.body, label %for.cond.cleanup.loopexit, !dbg !29
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="arm7tdmi" "target-features"="+strict-align" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !18, !19, !20}
!llvm.ident = !{!21}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/home/weimingz/llvm-build/release/community-tip")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !11)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7, !7, !10}
!7 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !8)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 32, align: 32)
!9 = !DIBasicType(name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!12, !13, !14, !15}
!12 = !DILocalVariable(name: "a", arg: 1, scope: !4, file: !1, line: 1, type: !7)
!13 = !DILocalVariable(name: "b", arg: 2, scope: !4, file: !1, line: 1, type: !7)
!14 = !DILocalVariable(name: "n", arg: 3, scope: !4, file: !1, line: 1, type: !10)
!15 = !DILocalVariable(name: "i", scope: !16, file: !1, line: 2, type: !10)
!16 = distinct !DILexicalBlock(scope: !4, file: !1, line: 2, column: 3)
!17 = !{i32 2, !"Dwarf Version", i32 4}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 1, !"wchar_size", i32 4}
!20 = !{i32 1, !"min_enum_size", i32 4}
!21 = !{!"clang version 3.8.0"}
!22 = !DIExpression()
!23 = !DILocation(line: 1, column: 27, scope: !4)
!24 = !DILocation(line: 1, column: 47, scope: !4)
!25 = !DILocation(line: 1, column: 54, scope: !4)
!26 = !DILocation(line: 2, column: 11, scope: !16)
!27 = !DILocation(line: 2, column: 20, scope: !28)
!28 = distinct !DILexicalBlock(scope: !16, file: !1, line: 2, column: 3)
!29 = !DILocation(line: 2, column: 3, scope: !16)
!30 = !DILocation(line: 3, column: 12, scope: !31)
!31 = distinct !DILexicalBlock(scope: !28, file: !1, line: 2, column: 31)
!32 = !DILocation(line: 8, column: 1, scope: !4)
!33 = !{!34, !34, i64 0}
!34 = !{!"float", !35, i64 0}
!35 = !{!"omnipotent char", !36, i64 0}
!36 = !{!"Simple C/C++ TBAA"}
!37 = !DILocation(line: 3, column: 5, scope: !31)
!38 = !DILocation(line: 3, column: 10, scope: !31)
!39 = !DILocation(line: 4, column: 17, scope: !31)
!40 = !DILocation(line: 4, column: 14, scope: !31)
!41 = !DILocation(line: 4, column: 5, scope: !31)
!42 = !DILocation(line: 4, column: 12, scope: !31)
!43 = !DILocation(line: 5, column: 17, scope: !31)
!44 = !DILocation(line: 5, column: 14, scope: !31)
!45 = !DILocation(line: 5, column: 5, scope: !31)
!46 = !DILocation(line: 5, column: 12, scope: !31)
!47 = !DILocation(line: 6, column: 17, scope: !31)
!48 = !DILocation(line: 6, column: 14, scope: !31)
!49 = !DILocation(line: 6, column: 5, scope: !31)
!50 = !DILocation(line: 6, column: 12, scope: !31)
!51 = !DILocation(line: 2, column: 26, scope: !28)
