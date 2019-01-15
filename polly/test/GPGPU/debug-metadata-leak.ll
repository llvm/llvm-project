; RUN: opt %loadPolly %s -polly-process-unprofitable -polly-codegen-ppcg -polly-acc-dump-kernel-ir \
; RUN: | FileCheck --check-prefix=KERNEL-IR %s

; REQUIRES: pollyacc

; KERNEL-IR: define ptx_kernel void @FUNC_vec_add_1_SCOP_0_KERNEL_0(i8 addrspace(1)* %MemRef_arr, i32 %N) #0 {

; The instruction marked <<<LeakyInst>>> is copied into the GPUModule,
; with changes only to the parameters to access data on the device instead of
; the host, i.e., MemRef_arr becomes polly.access.cast.MemRef_arr. Since the
; instruction is annotated with a DILocation, copying the instruction also copies
; the metadata into the GPUModule. This stops codegenerating the ptx_kernel by
; failing the verification of the Module in GPUNodeBuilder::finalize, due to the
; copied DICompileUnit not being listed in a llvm.dbg.cu which was neither copied
; nor created.
;
; https://reviews.llvm.org/D35630 removes this debug metadata before the
; instruction is copied to the GPUModule.
; 
; vec_add_1.c:
;      void vec_add_1(int N, int arr[N]) {
;        int i=0;
;        for( i=0 ; i<N ; i++) arr[i] += 1;
;      }
;
source_filename = "vec_add_1.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @vec_add_1(i32 %N, i32* %arr) !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 %N, i64 0, metadata !13, metadata !16), !dbg !17
  call void @llvm.dbg.value(metadata i32* %arr, i64 0, metadata !14, metadata !16), !dbg !18
  call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !15, metadata !16), !dbg !19
  %tmp = sext i32 %N to i64, !dbg !20
  br label %for.cond, !dbg !20

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  call void @llvm.dbg.value(metadata i32 undef, i64 0, metadata !15, metadata !16), !dbg !19
  %cmp = icmp slt i64 %indvars.iv, %tmp, !dbg !22
  br i1 %cmp, label %for.body, label %for.end, !dbg !24

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %arr, i64 %indvars.iv, !dbg !25
  %tmp1 = load i32, i32* %arrayidx, align 4, !dbg !26, !tbaa !27
  %add = add nsw i32 %tmp1, 1, !dbg !26    ;   <<<LeakyInst>>>
  store i32 %add, i32* %arrayidx, align 4, !dbg !26, !tbaa !27
  br label %for.inc, !dbg !25

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !31
  call void @llvm.dbg.value(metadata !2, i64 0, metadata !15, metadata !16), !dbg !19
  br label %for.cond, !dbg !32, !llvm.loop !33

for.end:                                          ; preds = %for.cond
  ret void, !dbg !35
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata)


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 5.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "vec_add_1.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 5.0.0"}
!7 = distinct !DISubprogram(name: "vec_add_1", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!12 = !{!13, !14, !15}
!13 = !DILocalVariable(name: "N", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!14 = !DILocalVariable(name: "arr", arg: 2, scope: !7, file: !1, line: 1, type: !11)
!15 = !DILocalVariable(name: "i", scope: !7, file: !1, line: 2, type: !10)
!16 = !DIExpression()
!17 = !DILocation(line: 1, column: 20, scope: !7)
!18 = !DILocation(line: 1, column: 27, scope: !7)
!19 = !DILocation(line: 2, column: 7, scope: !7)
!20 = !DILocation(line: 3, column: 8, scope: !21)
!21 = distinct !DILexicalBlock(scope: !7, file: !1, line: 3, column: 3)
!22 = !DILocation(line: 3, column: 15, scope: !23)
!23 = distinct !DILexicalBlock(scope: !21, file: !1, line: 3, column: 3)
!24 = !DILocation(line: 3, column: 3, scope: !21)
!25 = !DILocation(line: 3, column: 25, scope: !23)
!26 = !DILocation(line: 3, column: 32, scope: !23)
!27 = !{!28, !28, i64 0}
!28 = !{!"int", !29, i64 0}
!29 = !{!"omnipotent char", !30, i64 0}
!30 = !{!"Simple C/C++ TBAA"}
!31 = !DILocation(line: 3, column: 21, scope: !23)
!32 = !DILocation(line: 3, column: 3, scope: !23)
!33 = distinct !{!33, !24, !34}
!34 = !DILocation(line: 3, column: 35, scope: !21)
!35 = !DILocation(line: 4, column: 1, scope: !7)
