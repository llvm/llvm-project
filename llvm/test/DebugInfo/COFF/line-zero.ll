; RUN: llc < %s | FileCheck %s

; C++ source to regenerate:
; int main() {
;   volatile int x;
;   x = 1;
; #line 0
;   x = 2;
; #line 7
;   x = 3;
; }


; CHECK-LABEL: main:                                   # @main
; CHECK:         .cv_loc 0 1 1 0                 # t.cpp:1:0
; CHECK:         .cv_loc 0 1 3 0                 # t.cpp:3:0
; CHECK:         movl    $1, 4(%rsp)
; CHECK-NOT: .cv_loc {{.*}} t.cpp:0:0
; CHECK:         movl    $2, 4(%rsp)
; CHECK:         .cv_loc 0 1 7 0                 # t.cpp:7:0
; CHECK:         movl    $3, 4(%rsp)
; CHECK:         .cv_loc 0 1 8 0                 # t.cpp:8:0
; CHECK:         xorl    %eax, %eax
; CHECK:         retq

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.22.27905"

; Function Attrs: norecurse nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #0 !dbg !8 {
entry:
  %x = alloca i32, align 4
  %x.0.x.0..sroa_cast = bitcast i32* %x to i8*, !dbg !15
  call void @llvm.dbg.declare(metadata i32* %x, metadata !13, metadata !DIExpression()), !dbg !15
  store volatile i32 1, i32* %x, align 4, !dbg !16, !tbaa !17
  store volatile i32 2, i32* %x, align 4, !dbg !21, !tbaa !17
  store volatile i32 3, i32* %x, align 4, !dbg !22, !tbaa !17
  ret i32 0, !dbg !23
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

attributes #0 = { norecurse nounwind uwtable }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "t.cpp", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "8b6d53b166e6fa660f115eff7beedf3b")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 10.0.0"}
!8 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DILocalVariable(name: "x", scope: !8, file: !1, line: 2, type: !14)
!14 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !11)
!15 = !DILocation(line: 2, scope: !8)
!16 = !DILocation(line: 3, scope: !8)
!17 = !{!18, !18, i64 0}
!18 = !{!"int", !19, i64 0}
!19 = !{!"omnipotent char", !20, i64 0}
!20 = !{!"Simple C++ TBAA"}
!21 = !DILocation(line: 0, scope: !8)
!22 = !DILocation(line: 7, scope: !8)
!23 = !DILocation(line: 8, scope: !8)
