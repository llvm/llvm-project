; RUN: llc -x86-discriminate-memops  < %s | FileCheck %s
; RUN: llc -x86-discriminate-memops  -x86-bypass-prefetch-instructions=0 < %s | FileCheck %s -check-prefix=NOBYPASS
;
; original source, compiled with -O3 -gmlt -fdebug-info-for-profiling:
; int sum(int* arr, int pos1, int pos2) {
;   return arr[pos1] + arr[pos2];
; }
;
; ModuleID = 'test.cc'
source_filename = "test.cc"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.prefetch(i8 *, i32, i32, i32)
; Function Attrs: norecurse nounwind readonly uwtable
define i32 @sum(i32* %arr, i32 %pos1, i32 %pos2) !dbg !7 {
entry:
  %idxprom = sext i32 %pos1 to i64, !dbg !9
  %arrayidx = getelementptr inbounds i32, i32* %arr, i64 %idxprom, !dbg !9
  %0 = load i32, i32* %arrayidx, align 4, !dbg !9, !tbaa !10
  %idxprom1 = sext i32 %pos2 to i64, !dbg !14
  %arrayidx2 = getelementptr inbounds i32, i32* %arr, i64 %idxprom1, !dbg !14
  %addr = bitcast i32* %arrayidx2 to i8*
  call void @llvm.prefetch(i8* %addr, i32 0, i32 3, i32 1)
  %1 = load i32, i32* %arrayidx2, align 4, !dbg !14, !tbaa !10
  %add = add nsw i32 %1, %0, !dbg !15
  ret i32 %add, !dbg !16
}

attributes #0 = { "target-cpu"="x86-64" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, debugInfoForProfiling: true)
!1 = !DIFile(filename: "test.cc", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 7.0.0 (trunk 322155) (llvm/trunk 322159)"}
!7 = distinct !DISubprogram(name: "sum", linkageName: "sum", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 2, column: 10, scope: !7)
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C++ TBAA"}
!14 = !DILocation(line: 2, column: 22, scope: !7)
!15 = !DILocation(line: 2, column: 20, scope: !7)
!16 = !DILocation(line: 2, column: 3, scope: !7)

;CHECK-LABEL: sum:
;CHECK:       # %bb.0:
;CHECK:       prefetcht0 (%rdi,%rax,4)
;CHECK-NEXT:  movl (%rdi,%rax,4), %eax
;CHECK-NEXT:  .loc 1 2 20 discriminator 2  # test.cc:2:20
;CHECK-NEXT:  addl (%rdi,%rcx,4), %eax
;CHECK-NEXT:  .loc 1 2 3                   # test.cc:2:3

;NOBYPASS-LABEL: sum:
;NOBYPASS:       # %bb.0:
;NOBYPASS:       prefetcht0 (%rdi,%rax,4)
;NOBYPASS-NEXT: .loc 1 2 22
;NOBYPASS-NEXT:  movl (%rdi,%rax,4), %eax
;NOBYPASS-NEXT:  .loc 1 2 20 {{.*}} discriminator 2  # test.cc:2:20
;NOBYPASS-NEXT:  addl (%rdi,%rcx,4), %eax
;NOBYPASS-NEXT:  .loc 1 2 3                   # test.cc:2:3
