; RUN: llc < %s -march=hexagon -mcpu=hexagonv4 -O2 -disable-lsr | FileCheck %s
; ModuleID = 'hwloop-dbg.o'
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

define void @foo(i32* nocapture %a, i32* nocapture %b) nounwind !dbg !5 {
entry:
  tail call void @llvm.dbg.value(metadata i32* %a, i64 0, metadata !13, metadata !DIExpression()), !dbg !17
  tail call void @llvm.dbg.value(metadata i32* %b, i64 0, metadata !14, metadata !DIExpression()), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !15, metadata !DIExpression()), !dbg !19
  br label %for.body, !dbg !19

for.body:                                         ; preds = %for.body, %entry
; CHECK:     loop0(
; CHECK-NOT: add({{r[0-9]*}}, #
; CHECK:     endloop0
  %arrayidx.phi = phi i32* [ %a, %entry ], [ %arrayidx.inc, %for.body ]
  %i.02 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %b.addr.01 = phi i32* [ %b, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i32, i32* %b.addr.01, i32 1, !dbg !21
  tail call void @llvm.dbg.value(metadata i32* %incdec.ptr, i64 0, metadata !14, metadata !DIExpression()), !dbg !21
  %0 = load i32, i32* %b.addr.01, align 4, !dbg !21
  store i32 %0, i32* %arrayidx.phi, align 4, !dbg !21
  %inc = add nsw i32 %i.02, 1, !dbg !26
  tail call void @llvm.dbg.value(metadata i32 %inc, i64 0, metadata !15, metadata !DIExpression()), !dbg !26
  %exitcond = icmp eq i32 %inc, 10, !dbg !19
  %arrayidx.inc = getelementptr i32, i32* %arrayidx.phi, i32 1
  br i1 %exitcond, label %for.end, label %for.body, !dbg !19

for.end:                                          ; preds = %for.body
  ret void, !dbg !27
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!29}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "QuIC LLVM Hexagon Clang version 6.1-pre-unknown, (git://git-hexagon-aus.quicinc.com/llvm/clang-mainline.git e9382867661454cdf44addb39430741578e9765c) (llvm/llvm-mainline.git 36412bb1fcf03ed426d4437b41198bae066675ac)", isOptimized: true, emissionKind: FullDebug, file: !28, enums: !2, retainedTypes: !2, globals: !2)
!2 = !{}
!5 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 1, file: !28, scope: null, type: !7, retainedNodes: !11)
!6 = !DIFile(filename: "hwloop-dbg.c", directory: "/usr2/kparzysz/s.hex/t")
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9, !9}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, baseType: !10)
!10 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!13, !14, !15}
!13 = !DILocalVariable(name: "a", line: 1, arg: 1, scope: !5, file: !6, type: !9)
!14 = !DILocalVariable(name: "b", line: 1, arg: 2, scope: !5, file: !6, type: !9)
!15 = !DILocalVariable(name: "i", line: 2, scope: !16, file: !6, type: !10)
!16 = distinct !DILexicalBlock(line: 1, column: 26, file: !28, scope: !5)
!17 = !DILocation(line: 1, column: 15, scope: !5)
!18 = !DILocation(line: 1, column: 23, scope: !5)
!19 = !DILocation(line: 3, column: 8, scope: !20)
!20 = distinct !DILexicalBlock(line: 3, column: 3, file: !28, scope: !16)
!21 = !DILocation(line: 4, column: 5, scope: !22)
!22 = distinct !DILexicalBlock(line: 3, column: 28, file: !28, scope: !20)
!26 = !DILocation(line: 3, column: 23, scope: !20)
!27 = !DILocation(line: 6, column: 1, scope: !16)
!28 = !DIFile(filename: "hwloop-dbg.c", directory: "/usr2/kparzysz/s.hex/t")
!29 = !{i32 1, !"Debug Info Version", i32 3}
!30 = !{i32 0}
