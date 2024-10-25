; RUN: opt -passes=ipsccp -funcspec-for-literal-constant=false %s -S -o - | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators -passes=ipsccp -funcspec-for-literal-constant=false %s -S -o - | FileCheck %s

;; Check the dbg.assign DIAssignID operand gets remapped after cloning.

; CHECK: %tmp = alloca [4096 x i32], i32 0, align 16, !DIAssignID ![[ID1:[0-9]+]]
; CHECK-NEXT: #dbg_assign(i1 undef, !{{.*}}, !DIExpression(), ![[ID1]], ptr %tmp, !DIExpression(),
;
; CHECK: %tmp = alloca [4096 x i32], i32 0, align 16, !DIAssignID ![[ID2:[0-9]+]]
; CHECK-NEXT: #dbg_assign(i1 undef, !{{.*}}, !DIExpression(), ![[ID2]], ptr %tmp, !DIExpression(),

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata)

define void @inv_txfm_add_dct_dct_4x4_c() {
entry:
  call void @inv_txfm_add_c(ptr @dav1d_inv_dct4_1d_c)
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)

; Function Attrs: noinline
define void @inv_txfm_add_c(ptr %first_1d_fn) #2 {
entry:
  %tmp = alloca [4096 x i32], i32 0, align 16, !DIAssignID !5
  tail call void @llvm.dbg.assign(metadata i1 undef, metadata !6, metadata !DIExpression(), metadata !5, metadata ptr %tmp, metadata !DIExpression()), !dbg !16
  call void @llvm.memset.p0.i64(ptr %tmp, i8 0, i64 0, i1 false), !DIAssignID !17
  call void %first_1d_fn(ptr null, i64 0, i32 0, i32 0)
  ret void
}

declare void @dav1d_inv_dct4_1d_c(ptr, i64, i32, i32)

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

attributes #2 = { noinline }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, splitDebugInlining: false, nameTableKind: GNU)
!1 = !DIFile(filename: "itx_tmpl.c", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!5 = distinct !DIAssignID()
!6 = !DILocalVariable(name: "tmp", scope: !7, file: !1, line: 78, type: !10)
!7 = distinct !DISubprogram(name: "inv_txfm_add_c", scope: !1, file: !1, line: 41, type: !8, scopeLine: 45, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!8 = distinct !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 131072, elements: !2)
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "int32_t", file: !12, line: 26, baseType: !13)
!12 = !DIFile(filename: "stdint-intn.h", directory: ".")
!13 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int32_t", file: !14, line: 41, baseType: !15)
!14 = !DIFile(filename: "types.h", directory: ".")
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !DILocation(line: 0, scope: !7)
!17 = distinct !DIAssignID()
