; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

; Function Attrs: nounwind
define dso_local void @test(float* nocapture readonly %Fptr, <4 x float>* nocapture %Vptr) local_unnamed_addr #0 !dbg !10 {
; CHECK-LABEL: test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:          #DEBUG_VALUE: test:Fptr <- $x3
; CHECK-NEXT:          #DEBUG_VALUE: test:Vptr <- $x4
; CHECK-NEXT:          addis 5, 2, .LCPI0_0@toc@ha
; CHECK-NEXT:  .Ltmp0:
; CHECK-NEXT:          .loc    1 2 38 prologue_end
; CHECK-NEXT:          lfsx 0, 0, 3
; CHECK-NEXT:          addis 3, 2, .LCPI0_1@toc@ha
; CHECK-NEXT:  .Ltmp1:
; CHECK-NEXT:          #DEBUG_VALUE: test:Fptr <- $x3
; CHECK-NEXT:          .loc    1 0 38 is_stmt 0
; CHECK-NEXT:          lfs 1, .LCPI0_0@toc@l(5)
; CHECK-NEXT:          lfd 2, .LCPI0_1@toc@l(3)
; CHECK-NEXT:          .loc    1 2 27
; CHECK-NEXT:          xssubdp 1, 1, 0
; CHECK-NEXT:          .loc    1 2 45
; CHECK-NEXT:          xsadddp 1, 1, 2
; CHECK-NEXT:  .Ltmp2:
; CHECK-NEXT:          #DEBUG_VALUE: test:Val <- undef
; CHECK-NEXT:          .loc    1 0 45
; CHECK-NEXT:          xxlxor 2, 2, 2
; CHECK-NEXT:          .loc    1 3 26 is_stmt 1
; CHECK-NEXT:          xxmrghd 0, 0, 2
; CHECK-NEXT:          xxmrghd 1, 2, 1
; CHECK-NEXT:          xvcvdpsp 34, 0
; CHECK-NEXT:          xvcvdpsp 35, 1
; CHECK-NEXT:          vmrgew 2, 2, 3
; CHECK-NEXT:          #DEBUG_VALUE: test:Vptr <- $x4
; CHECK-NEXT:          .loc    1 3 9 is_stmt 0
; CHECK-NEXT:          stvx 2, 0, 4
; CHECK-NEXT:          .loc    1 4 1 is_stmt 1
; CHECK-NEXT:          blr
entry:
  call void @llvm.dbg.value(metadata float* %Fptr, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata <4 x float>* %Vptr, metadata !20, metadata !DIExpression()), !dbg !22
  %0 = load float, float* %Fptr, align 4, !dbg !23, !tbaa !24
  %conv = fpext float %0 to double, !dbg !28
  %sub = fsub double 1.000000e+00, %conv, !dbg !29
  %sub1 = fadd double %sub, -4.300000e+00, !dbg !30
  %conv2 = fptrunc double %sub1 to float, !dbg !31
  call void @llvm.dbg.value(metadata float %conv2, metadata !21, metadata !DIExpression()), !dbg !22
  %vecinit4 = insertelement <4 x float> <float poison, float 0.000000e+00, float 0.000000e+00, float poison>, float %conv2, i32 0, !dbg !32
  %vecinit5 = insertelement <4 x float> %vecinit4, float %0, i32 3, !dbg !32
  store <4 x float> %vecinit5, <4 x float>* %Vptr, align 16, !dbg !33, !tbaa !34
  ret void, !dbg !35
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{}
!3 = !{!4, !5}
!4 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!5 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{!"clang version 12.0.0"}
!10 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !18)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13, !14}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, size: 128, flags: DIFlagVector, elements: !16)
!16 = !{!17}
!17 = !DISubrange(count: 4)
!18 = !{!19, !20, !21}
!19 = !DILocalVariable(name: "Fptr", arg: 1, scope: !10, file: !1, line: 1, type: !13)
!20 = !DILocalVariable(name: "Vptr", arg: 2, scope: !10, file: !1, line: 1, type: !14)
!21 = !DILocalVariable(name: "Val", scope: !10, file: !1, line: 2, type: !4)
!22 = !DILocation(line: 0, scope: !10)
!23 = !DILocation(line: 2, column: 38, scope: !10)
!24 = !{!25, !25, i64 0}
!25 = !{!"float", !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C/C++ TBAA"}
!28 = !DILocation(line: 2, column: 29, scope: !10)
!29 = !DILocation(line: 2, column: 27, scope: !10)
!30 = !DILocation(line: 2, column: 45, scope: !10)
!31 = !DILocation(line: 2, column: 15, scope: !10)
!32 = !DILocation(line: 3, column: 26, scope: !10)
!33 = !DILocation(line: 3, column: 9, scope: !10)
!34 = !{!26, !26, i64 0}
!35 = !DILocation(line: 4, column: 1, scope: !10)
