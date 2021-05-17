; RUN: opt -S < %s -loop-vectorize -force-vector-interleave=1 -force-vector-width=2 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Make sure we are preserving debug info in the vectorized code.

; CHECK: for.body.lr.ph
; CHECK:   min.iters.check = icmp ult i64 {{.*}}, 2, !dbg !{{[0-9]+}}
; CHECK: vector.body
; CHECK:   index {{.*}}, !dbg ![[LOC:[0-9]+]]
; CHECK:   getelementptr inbounds i32, i32* %a, {{.*}}, !dbg ![[LOC]]
; CHECK:   load <2 x i32>, <2 x i32>* {{.*}}, !dbg ![[LOC]]
; CHECK:   add <2 x i32> {{.*}}, !dbg ![[LOC]]
; CHECK:   add i64 %index, 2, !dbg ![[LOC]]
; CHECK:   icmp eq i64 %index.next, %n.vec, !dbg ![[LOC]]
; CHECK: middle.block
; CHECK:   call i32 @llvm.vector.reduce.add.v2i32(<2 x i32> %{{.*}}), !dbg ![[BR_LOC:[0-9]+]]
; CHECK: for.body
; CHECK: br i1{{.*}}, label %for.body,{{.*}}, !dbg ![[BR_LOC]],
; CHECK: ![[BR_LOC]] = !DILocation(line: 5,

define i32 @f(i32* nocapture %a, i32 %size) #0 !dbg !4 {
entry:
  call void @llvm.dbg.value(metadata i32* %a, metadata !13, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata i32 %size, metadata !14, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 0, metadata !16, metadata !DIExpression()), !dbg !21
  %cmp4 = icmp eq i32 %size, 0, !dbg !21
  br i1 %cmp4, label %for.end, label %for.body.lr.ph, !dbg !21

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body, !dbg !21

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %sum.05 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv, !dbg !22
  %0 = load i32, i32* %arrayidx, align 4, !dbg !22
  %add = add i32 %0, %sum.05, !dbg !22
  %indvars.iv.next = add i64 %indvars.iv, 1, !dbg !22
  call void @llvm.dbg.value(metadata !{null}, metadata !16, metadata !DIExpression()), !dbg !22
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !22
  %exitcond = icmp ne i32 %lftr.wideiv, %size, !dbg !21
  br i1 %exitcond, label %for.body, label %for.cond.for.end_crit_edge, !dbg !21

for.cond.for.end_crit_edge:                       ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  call void @llvm.dbg.value(metadata i32 %add.lcssa, metadata !15, metadata !DIExpression()), !dbg !22
  br label %for.end, !dbg !21

for.end:                                          ; preds = %entry, %for.cond.for.end_crit_edge
  %sum.0.lcssa = phi i32 [ %add.lcssa, %for.cond.for.end_crit_edge ], [ 0, %entry ]
  ret i32 %sum.0.lcssa, !dbg !26
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind readonly ssp uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18, !27}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.4 (trunk 185038) (llvm/trunk 185097)", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "-", directory: "/Volumes/Data/backedup/dev/os/llvm/debug")
!2 = !{}
!4 = distinct !DISubprogram(name: "f", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 3, file: !5, scope: !6, type: !7, retainedNodes: !12)
!5 = !DIFile(filename: "<stdin>", directory: "/Volumes/Data/backedup/dev/os/llvm/debug")
!6 = !DIFile(filename: "<stdin>", directory: "/Volumes/Data/backedup/dev/os/llvm/debug")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !10, !11}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !9)
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!12 = !{!13, !14, !15, !16}
!13 = !DILocalVariable(name: "a", line: 3, arg: 1, scope: !4, file: !6, type: !10)
!14 = !DILocalVariable(name: "size", line: 3, arg: 2, scope: !4, file: !6, type: !11)
!15 = !DILocalVariable(name: "sum", line: 4, scope: !4, file: !6, type: !11)
!16 = !DILocalVariable(name: "i", line: 5, scope: !17, file: !6, type: !11)
!17 = distinct !DILexicalBlock(line: 5, column: 0, file: !5, scope: !4)
!18 = !{i32 2, !"Dwarf Version", i32 3}
!19 = !DILocation(line: 3, scope: !4)
!20 = !DILocation(line: 4, scope: !4)
!21 = !DILocation(line: 5, scope: !17)
!22 = !DILocation(line: 6, scope: !17)
!26 = !DILocation(line: 7, scope: !4)
!27 = !{i32 1, !"Debug Info Version", i32 3}
