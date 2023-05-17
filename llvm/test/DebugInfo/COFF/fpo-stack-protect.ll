; RUN: llc < %s -experimental-debug-variable-locations=true | FileCheck %s

; C source:
; void escape(int *);
; int ssp(int a) {
;   int arr[4] = {a, a, a, a};
;   escape(&arr[0]);
;   return a;
; }

; CHECK: _ssp:                                   # @ssp
; CHECK:         .cv_fpo_proc    _ssp 4
; CHECK:         pushl   %esi
; CHECK:         .cv_fpo_pushreg %esi
; CHECK:         subl    $20, %esp
; CHECK:         .cv_fpo_stackalloc      20
; CHECK:         .cv_fpo_endprologue
; CHECK:         ___security_cookie

; CHECK:         movl    28(%esp), %esi
; CHECK:         movl    %esi, {{[0-9]*}}(%esp)
; CHECK:         movl    %esi, {{[0-9]*}}(%esp)
; CHECK:         movl    %esi, {{[0-9]*}}(%esp)
; CHECK:         movl    %esi, {{[0-9]*}}(%esp)

; CHECK:         calll   _escape
; CHECK:         calll   @__security_check_cookie@4

; CHECK:         movl    %esi, %eax
; CHECK:         addl    $20, %esp
; CHECK:         popl    %esi
; CHECK:         retl
; CHECK: Ltmp4:
; CHECK:         .cv_fpo_endproc

; ModuleID = 't.c'
source_filename = "t.c"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc19.11.25508"

; Function Attrs: nounwind sspstrong
define i32 @ssp(i32 returned %a) local_unnamed_addr #0 !dbg !8 {
entry:
  %arr = alloca [4 x i32], align 4
  tail call void @llvm.dbg.value(metadata i32 %a, metadata !13, metadata !DIExpression()), !dbg !18
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %arr) #4, !dbg !19
  tail call void @llvm.dbg.declare(metadata ptr %arr, metadata !14, metadata !DIExpression()), !dbg !20
  store i32 %a, ptr %arr, align 4, !dbg !21, !tbaa !22
  %arrayinit.element = getelementptr inbounds [4 x i32], ptr %arr, i32 0, i32 1, !dbg !21
  store i32 %a, ptr %arrayinit.element, align 4, !dbg !21, !tbaa !22
  %arrayinit.element1 = getelementptr inbounds [4 x i32], ptr %arr, i32 0, i32 2, !dbg !21
  store i32 %a, ptr %arrayinit.element1, align 4, !dbg !21, !tbaa !22
  %arrayinit.element2 = getelementptr inbounds [4 x i32], ptr %arr, i32 0, i32 3, !dbg !21
  store i32 %a, ptr %arrayinit.element2, align 4, !dbg !21, !tbaa !22
  call void @escape(ptr nonnull %arr) #4, !dbg !26
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %arr) #4, !dbg !27
  ret i32 %a, !dbg !28
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #2

declare void @escape(ptr) local_unnamed_addr #3

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #2

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind sspstrong "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.c", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "df0c1a43acd19a1255d45a3f2802cf9f")
!2 = !{}
!3 = !{i32 1, !"NumRegisterParameters", i32 0}
!4 = !{i32 2, !"CodeView", i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 2}
!7 = !{!"clang version 6.0.0 "}
!8 = distinct !DISubprogram(name: "ssp", scope: !1, file: !1, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13, !14}
!13 = !DILocalVariable(name: "a", arg: 1, scope: !8, file: !1, line: 2, type: !11)
!14 = !DILocalVariable(name: "arr", scope: !8, file: !1, line: 3, type: !15)
!15 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 128, elements: !16)
!16 = !{!17}
!17 = !DISubrange(count: 4)
!18 = !DILocation(line: 2, column: 13, scope: !8)
!19 = !DILocation(line: 3, column: 3, scope: !8)
!20 = !DILocation(line: 3, column: 7, scope: !8)
!21 = !DILocation(line: 3, column: 16, scope: !8)
!22 = !{!23, !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C/C++ TBAA"}
!26 = !DILocation(line: 4, column: 3, scope: !8)
!27 = !DILocation(line: 6, column: 1, scope: !8)
!28 = !DILocation(line: 5, column: 3, scope: !8)
