; RUN: opt  -instcombine %s -S | FileCheck %s
;
; Generate me from:
; clang -cc1 -triple thumbv7-apple-ios7.0.0 -S -target-abi apcs-gnu -gdwarf-2 -Os test.c -o test.ll -emit-llvm
; void run(float r)
; {
;   int count = r;
;   float vla[count];
;   vla[0] = r;
;   for (int i = 0; i < count; i++)
;     vla[i] /= r;
; }
; rdar://problem/15464571
;
; ModuleID = 'test.c'
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios8.0.0"

; Function Attrs: nounwind optsize readnone
define void @run(float %r) #0 !dbg !4 {
entry:
  tail call void @llvm.dbg.declare(metadata float %r, metadata !11, metadata !DIExpression()), !dbg !22
  %conv = fptosi float %r to i32, !dbg !23
  tail call void @llvm.dbg.declare(metadata i32 %conv, metadata !12, metadata !DIExpression()), !dbg !23
  %vla = alloca float, i32 %conv, align 4, !dbg !24
  tail call void @llvm.dbg.declare(metadata float* %vla, metadata !14, metadata !DIExpression(DW_OP_deref)), !dbg !24
; The VLA alloca should be described by a dbg.declare:
; CHECK: call void @llvm.dbg.declare(metadata float* %vla, metadata ![[VLA:.*]], metadata {{.*}})
; The VLA alloca and following store into the array should not be lowered to like this:
; CHECK-NOT:  call void @llvm.dbg.value(metadata float %r, metadata ![[VLA]])
; the backend interprets this as "vla has the location of %r".
  store float %r, float* %vla, align 4, !dbg !25, !tbaa !26
  tail call void @llvm.dbg.value(metadata i32 0, metadata !18, metadata !DIExpression()), !dbg !30
  %cmp8 = icmp sgt i32 %conv, 0, !dbg !30
  br i1 %cmp8, label %for.body, label %for.end, !dbg !30

for.body:                                         ; preds = %entry, %for.body.for.body_crit_edge
  %0 = phi float [ %.pre, %for.body.for.body_crit_edge ], [ %r, %entry ]
  %i.09 = phi i32 [ %inc, %for.body.for.body_crit_edge ], [ 0, %entry ]
  %arrayidx2 = getelementptr inbounds float, float* %vla, i32 %i.09, !dbg !31
  %div = fdiv float %0, %r, !dbg !31
  store float %div, float* %arrayidx2, align 4, !dbg !31, !tbaa !26
  %inc = add nsw i32 %i.09, 1, !dbg !30
  tail call void @llvm.dbg.value(metadata i32 %inc, metadata !18, metadata !DIExpression()), !dbg !30
  %exitcond = icmp eq i32 %inc, %conv, !dbg !30
  br i1 %exitcond, label %for.end, label %for.body.for.body_crit_edge, !dbg !30

for.body.for.body_crit_edge:                      ; preds = %for.body
  %arrayidx2.phi.trans.insert = getelementptr inbounds float, float* %vla, i32 %inc
  %.pre = load float, float* %arrayidx2.phi.trans.insert, align 4, !dbg !31, !tbaa !26
  br label %for.body, !dbg !30

for.end:                                          ; preds = %for.body, %entry
  ret void, !dbg !32
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind optsize readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20, !33}
!llvm.ident = !{!21}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.4 ", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "<unknown>", directory: "/Volumes/Data/radar/15464571")
!2 = !{}
!4 = distinct !DISubprogram(name: "run", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 2, file: !5, scope: !6, type: !7, variables: !10)
!5 = !DIFile(filename: "test.c", directory: "/Volumes/Data/radar/15464571")
!6 = !DIFile(filename: "test.c", directory: "/Volumes/Data/radar/15464571")
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!10 = !{!11, !12, !14, !18}
!11 = !DILocalVariable(name: "r", line: 1, arg: 1, scope: !4, file: !6, type: !9)
!12 = !DILocalVariable(name: "count", line: 3, scope: !4, file: !6, type: !13)
!13 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !DILocalVariable(name: "vla", line: 4, scope: !4, file: !6, type: !15)
!15 = !DICompositeType(tag: DW_TAG_array_type, align: 32, baseType: !9, elements: !16)
!16 = !{!17}
!17 = !DISubrange(count: -1)
!18 = !DILocalVariable(name: "i", line: 6, scope: !19, file: !6, type: !13)
!19 = distinct !DILexicalBlock(line: 6, column: 0, file: !5, scope: !4)
!20 = !{i32 2, !"Dwarf Version", i32 2}
!21 = !{!"clang version 3.4 "}
!22 = !DILocation(line: 1, scope: !4)
!23 = !DILocation(line: 3, scope: !4)
!24 = !DILocation(line: 4, scope: !4)
!25 = !DILocation(line: 5, scope: !4)
!26 = !{!27, !27, i64 0}
!27 = !{!"float", !28, i64 0}
!28 = !{!"omnipotent char", !29, i64 0}
!29 = !{!"Simple C/C++ TBAA"}
!30 = !DILocation(line: 6, scope: !19)
!31 = !DILocation(line: 7, scope: !19)
!32 = !DILocation(line: 8, scope: !4)
!33 = !{i32 1, !"Debug Info Version", i32 3}
