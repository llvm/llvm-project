; RUN: llc < %s | FileCheck %s

; C source:
; void usethings(double *, void *p);
; int realign_and_alloca(int n) {
;   double d = 0;
;   void *p = __builtin_alloca(n);
;   usethings(&d, p);
;   return 0;
; }

; CHECK: _realign_and_alloca:                    # @realign_and_alloca
; CHECK:         .cv_fpo_proc    _realign_and_alloca 4
; CHECK:         pushl   %ebp
; CHECK:         .cv_fpo_pushreg %ebp
; CHECK:         movl    %esp, %ebp
; CHECK:         .cv_fpo_setframe        %ebp
; CHECK:         pushl   %esi
; CHECK:         .cv_fpo_pushreg %esi
;       We don't seem to need to describe this AND because at this point CSRs
;       are stored relative to EBP, but it's suspicious.
; CHECK:         andl    $-16, %esp
; CHECK:         subl    $32, %esp
; CHECK:         .cv_fpo_stackalloc      32
; CHECK:         .cv_fpo_endprologue
; CHECK:         movl    %esp, %esi
; CHECK:         leal    8(%esi),
; CHECK:         calll   _usethings
; CHECK:         addl    $8, %esp
; CHECK:         xorl    %eax, %eax
; CHECK:         leal    -4(%ebp), %esp
; CHECK:         popl    %esi
; CHECK:         popl    %ebp
; CHECK:         retl
; CHECK:         .cv_fpo_endproc


; ModuleID = 't.c'
source_filename = "t.c"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc19.11.25508"

; Function Attrs: nounwind
define i32 @realign_and_alloca(i32 %n) local_unnamed_addr #0 !dbg !8 {
entry:
  %d = alloca double, align 8
  tail call void @llvm.dbg.value(metadata i32 %n, metadata !13, metadata !DIExpression()), !dbg !18
  %0 = bitcast double* %d to i8*, !dbg !19
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #4, !dbg !19
  tail call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !14, metadata !DIExpression()), !dbg !20
  store double 0.000000e+00, double* %d, align 8, !dbg !20, !tbaa !21
  %1 = alloca i8, i32 %n, align 16, !dbg !25
  tail call void @llvm.dbg.value(metadata i8* %1, metadata !16, metadata !DIExpression()), !dbg !26
  tail call void @llvm.dbg.value(metadata double* %d, metadata !14, metadata !DIExpression()), !dbg !20
  call void @usethings(double* nonnull %d, i8* nonnull %1) #4, !dbg !27
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #4, !dbg !28
  ret i32 0, !dbg !29
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

declare void @usethings(double*, i8*) local_unnamed_addr #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone speculatable }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.c", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "cfdc2deff5dc50f95e287f877660d4dd")
!2 = !{}
!3 = !{i32 1, !"NumRegisterParameters", i32 0}
!4 = !{i32 2, !"CodeView", i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 2}
!7 = !{!"clang version 6.0.0 "}
!8 = distinct !DISubprogram(name: "realign_and_alloca", scope: !1, file: !1, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13, !14, !16}
!13 = !DILocalVariable(name: "n", arg: 1, scope: !8, file: !1, line: 2, type: !11)
!14 = !DILocalVariable(name: "d", scope: !8, file: !1, line: 3, type: !15)
!15 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!16 = !DILocalVariable(name: "p", scope: !8, file: !1, line: 4, type: !17)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 32)
!18 = !DILocation(line: 2, column: 28, scope: !8)
!19 = !DILocation(line: 3, column: 3, scope: !8)
!20 = !DILocation(line: 3, column: 10, scope: !8)
!21 = !{!22, !22, i64 0}
!22 = !{!"double", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C/C++ TBAA"}
!25 = !DILocation(line: 4, column: 13, scope: !8)
!26 = !DILocation(line: 4, column: 9, scope: !8)
!27 = !DILocation(line: 5, column: 3, scope: !8)
!28 = !DILocation(line: 7, column: 1, scope: !8)
!29 = !DILocation(line: 6, column: 3, scope: !8)
