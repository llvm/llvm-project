; RUN: llc -o %t.obj %s -filetype=obj
; RUN: llvm-pdbutil dump -symbols -types %t.obj | FileCheck %s
;
; The IR in this test derives from the following Fortran program
; with inlining enabled:
; program if_test
;   implicit none
;   integer, allocatable :: a
;   allocate(a)
;   call sub(a)
; contains
;   subroutine sub(aa)
;     implicit none
;     integer :: aa, bb
;     bb = 1
;     aa = bb
;   end subroutine sub
; end program if_test
;
; CHECK: [[proc_t:.*]] | LF_PROCEDURE
;
; CHECK: [[func_id_sub:.*]] | LF_FUNC_ID
; CHECK-NEXT: name = SUB, type = [[proc_t]], parent scope = <no type>
;
; CHECK: [[func_id_if_test:.*]] | LF_FUNC_ID
; CHECK-NEXT: name = IF_TEST, type = [[proc_t]], parent scope = <no type>
;
; CHECK: S_GPROC32_ID [size = {{.*}}] `IF_TEST`
; CHECK-NEXT: parent
; CHECK-NEXT: type = `[[func_id_if_test]] (IF_TEST)`
;
; ModuleID = 'tr1.f90'
source_filename = "tr1.f90"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

@"IF_TEST$A" = internal global i32* null, align 8, !dbg !0
@0 = internal unnamed_addr constant i32 65536
@1 = internal unnamed_addr constant i32 2

; Function Attrs: nounwind uwtable
define void @MAIN__() local_unnamed_addr #0 !dbg !2 {
alloca_0:
  %func_result = tail call i32 @for_set_fpe_(i32* nonnull @0) #4, !dbg !22
  %func_result2 = tail call i32 @for_set_reentrancy(i32* nonnull @1) #4, !dbg !22
  %func_result4 = tail call i32 @for_alloc_allocatable(i64 4, i8** bitcast (i32** @"IF_TEST$A" to i8**), i32 262144) #4, !dbg !23
  %"IF_TEST$A_fetch.1" = load i32*, i32** @"IF_TEST$A", align 8, !dbg !24, !tbaa !25
  call void @llvm.dbg.declare(metadata i32* %"IF_TEST$A_fetch.1", metadata !29, metadata !DIExpression()), !dbg !33
  call void @llvm.dbg.value(metadata i32 1, metadata !32, metadata !DIExpression()), !dbg !35
  store i32 1, i32* %"IF_TEST$A_fetch.1", align 1, !dbg !36, !tbaa !37, !alias.scope !41
  ret void, !dbg !44
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind uwtable willreturn writeonly
define void @IF_TEST_ip_SUB(i32* noalias nocapture dereferenceable(4) %AA) local_unnamed_addr #1 !dbg !30 {
alloca_1:
  call void @llvm.dbg.declare(metadata i32* %AA, metadata !29, metadata !DIExpression()), !dbg !45
  call void @llvm.dbg.value(metadata i32 1, metadata !32, metadata !DIExpression()), !dbg !46
  store i32 1, i32* %AA, align 1, !dbg !47, !tbaa !37
  ret void, !dbg !48
}

declare i32 @for_set_fpe_(i32* nocapture readonly) local_unnamed_addr

; Function Attrs: nofree
declare i32 @for_set_reentrancy(i32* nocapture readonly) local_unnamed_addr #2

; Function Attrs: nofree
declare i32 @for_alloc_allocatable(i64, i8** nocapture, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #3

; Function Attrs: mustprogress nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { nounwind uwtable "denormal-fp-math"="preserve_sign,preserve_sign" "frame-pointer"="none" "intel-lang"="fortran" "min-legal-vector-width"="0" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind uwtable willreturn writeonly "denormal-fp-math"="preserve_sign,preserve_sign" "frame-pointer"="none" "intel-lang"="fortran" "min-legal-vector-width"="0" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #2 = { nofree "intel-lang"="fortran" }
attributes #3 = { mustprogress nofree nosync nounwind readnone speculatable willreturn }
attributes #4 = { nounwind }

!llvm.module.flags = !{!11, !12, !13}
!llvm.dbg.cu = !{!6}
!omp_offload.info = !{}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "A", linkageName: "IF_TEST$A", scope: !2, file: !3, line: 11, type: !9, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "IF_TEST", linkageName: "MAIN__", scope: !3, file: !3, line: 9, type: !4, scopeLine: 9, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !6, retainedNodes: !8)
!3 = !DIFile(filename: "tr1.f90", directory: "d:\\iusers\\cchen15\\examples\\tests\\jr14335")
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !3, producer: "Intel(R) Fortran 22.0-1087", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !7, splitDebugInlining: false, nameTableKind: None)
!7 = !{!0}
!8 = !{}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = !DIBasicType(name: "INTEGER*4", size: 32, encoding: DW_ATE_signed)
!11 = !{i32 7, !"PIC Level", i32 2}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 2, !"CodeView", i32 1}
!22 = !DILocation(line: 9, scope: !2)
!23 = !DILocation(line: 12, scope: !2)
!24 = !DILocation(line: 14, scope: !2)
!25 = !{!26, !26, i64 0}
!26 = !{!"ifx$unique_sym$1", !27, i64 0}
!27 = !{!"Generic Fortran Symbol", !28, i64 0}
!28 = !{!"ifx$root$1$MAIN__"}
!29 = !DILocalVariable(name: "AA", arg: 1, scope: !30, file: !3, line: 17, type: !10)
!30 = distinct !DISubprogram(name: "SUB", linkageName: "IF_TEST_ip_SUB", scope: !2, file: !3, line: 17, type: !4, scopeLine: 17, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !31)
!31 = !{!29, !32}
!32 = !DILocalVariable(name: "BB", scope: !30, file: !3, line: 19, type: !10)
!33 = !DILocation(line: 17, scope: !30, inlinedAt: !34)
!34 = distinct !DILocation(line: 14, scope: !2)
!35 = !DILocation(line: 0, scope: !30, inlinedAt: !34)
!36 = !DILocation(line: 21, scope: !30, inlinedAt: !34)
!37 = !{!38, !38, i64 0}
!38 = !{!"ifx$unique_sym$3", !39, i64 0}
!39 = !{!"Generic Fortran Symbol", !40, i64 0}
!40 = !{!"ifx$root$2$IF_TEST_ip_SUB"}
!41 = !{!42}
!42 = distinct !{!42, !43, !"IF_TEST_ip_SUB: %AA"}
!43 = distinct !{!43, !"IF_TEST_ip_SUB"}
!44 = !DILocation(line: 16, scope: !2)
!45 = !DILocation(line: 17, scope: !30)
!46 = !DILocation(line: 0, scope: !30)
!47 = !DILocation(line: 21, scope: !30)
!48 = !DILocation(line: 22, scope: !30)
