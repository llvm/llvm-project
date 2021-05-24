; RUN: opt %s -scalarizer -scalarize-load-store -S | FileCheck %s
; RUN: opt %s -passes='function(scalarizer)' -scalarize-load-store -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @f1(<4 x i32>* nocapture %a, <4 x i32>* nocapture readonly %b, <4 x i32>* nocapture readonly %c) #0 !dbg !4 {
; CHECK: @f1(
; CHECK: %a.i0 = bitcast <4 x i32>* %a to i32*
; CHECK: %a.i1 = getelementptr i32, i32* %a.i0, i32 1
; CHECK: %a.i2 = getelementptr i32, i32* %a.i0, i32 2
; CHECK: %a.i3 = getelementptr i32, i32* %a.i0, i32 3
; CHECK: %c.i0 = bitcast <4 x i32>* %c to i32*
; CHECK: %c.i1 = getelementptr i32, i32* %c.i0, i32 1
; CHECK: %c.i2 = getelementptr i32, i32* %c.i0, i32 2
; CHECK: %c.i3 = getelementptr i32, i32* %c.i0, i32 3
; CHECK: %b.i0 = bitcast <4 x i32>* %b to i32*
; CHECK: %b.i1 = getelementptr i32, i32* %b.i0, i32 1
; CHECK: %b.i2 = getelementptr i32, i32* %b.i0, i32 2
; CHECK: %b.i3 = getelementptr i32, i32* %b.i0, i32 3
; CHECK: tail call void @llvm.dbg.value(metadata <4 x i32>* %a, metadata !{{[0-9]+}}, metadata {{.*}}), !dbg !{{[0-9]+}}
; CHECK: tail call void @llvm.dbg.value(metadata <4 x i32>* %b, metadata !{{[0-9]+}}, metadata {{.*}}), !dbg !{{[0-9]+}}
; CHECK: tail call void @llvm.dbg.value(metadata <4 x i32>* %c, metadata !{{[0-9]+}}, metadata {{.*}}), !dbg !{{[0-9]+}}
; CHECK: %bval.i0 = load i32, i32* %b.i0, align 16, !dbg ![[TAG1:[0-9]+]], !tbaa ![[TAG2:[0-9]+]]
; CHECK: %bval.i1 = load i32, i32* %b.i1, align 4, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: %bval.i2 = load i32, i32* %b.i2, align 8, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: %bval.i3 = load i32, i32* %b.i3, align 4, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: %cval.i0 = load i32, i32* %c.i0, align 16, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: %cval.i1 = load i32, i32* %c.i1, align 4, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: %cval.i2 = load i32, i32* %c.i2, align 8, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: %cval.i3 = load i32, i32* %c.i3, align 4, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: %add.i0 = add i32 %bval.i0, %cval.i0, !dbg ![[TAG1]]
; CHECK: %add.i1 = add i32 %bval.i1, %cval.i1, !dbg ![[TAG1]]
; CHECK: %add.i2 = add i32 %bval.i2, %cval.i2, !dbg ![[TAG1]]
; CHECK: %add.i3 = add i32 %bval.i3, %cval.i3, !dbg ![[TAG1]]
; CHECK: store i32 %add.i0, i32* %a.i0, align 16, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: store i32 %add.i1, i32* %a.i1, align 4, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: store i32 %add.i2, i32* %a.i2, align 8, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: store i32 %add.i3, i32* %a.i3, align 4, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: ret void
entry:
  tail call void @llvm.dbg.value(metadata <4 x i32>* %a, metadata !15, metadata !DIExpression()), !dbg !20
  tail call void @llvm.dbg.value(metadata <4 x i32>* %b, metadata !16, metadata !DIExpression()), !dbg !20
  tail call void @llvm.dbg.value(metadata <4 x i32>* %c, metadata !17, metadata !DIExpression()), !dbg !20
  %bval = load <4 x i32>, <4 x i32>* %b, align 16, !dbg !21, !tbaa !22
  %cval = load <4 x i32>, <4 x i32>* %c, align 16, !dbg !21, !tbaa !22
  %add = add <4 x i32> %bval, %cval, !dbg !21
  store <4 x i32> %add, <4 x i32>* %a, align 16, !dbg !21, !tbaa !22
  ret void, !dbg !25
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18, !26}
!llvm.ident = !{!19}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.4 (trunk 194134) (llvm/trunk 194126)", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "/tmp/add.c", directory: "/home/richards/llvm/build")
!2 = !{}
!4 = distinct !DISubprogram(name: "f1", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 4, file: !1, scope: !5, type: !6, retainedNodes: !14)
!5 = !DIFile(filename: "/tmp/add.c", directory: "/home/richards/llvm/build")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !8, !8}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !9)
!9 = !DIDerivedType(tag: DW_TAG_typedef, name: "V4SI", line: 1, file: !1, baseType: !10)
!10 = !DICompositeType(tag: DW_TAG_array_type, size: 128, align: 128, flags: DIFlagVector, baseType: !11, elements: !12)
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DISubrange(count: 4)
!14 = !{!15, !16, !17}
!15 = !DILocalVariable(name: "a", line: 3, arg: 1, scope: !4, file: !5, type: !8)
!16 = !DILocalVariable(name: "b", line: 3, arg: 2, scope: !4, file: !5, type: !8)
!17 = !DILocalVariable(name: "c", line: 3, arg: 3, scope: !4, file: !5, type: !8)
!18 = !{i32 2, !"Dwarf Version", i32 4}
!19 = !{!"clang version 3.4 (trunk 194134) (llvm/trunk 194126)"}
!20 = !DILocation(line: 3, scope: !4)
!21 = !DILocation(line: 5, scope: !4)
!22 = !{!23, !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C/C++ TBAA"}
!25 = !DILocation(line: 6, scope: !4)
!26 = !{i32 1, !"Debug Info Version", i32 3}
