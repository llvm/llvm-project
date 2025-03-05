; RUN: llc < %s -x86-discriminate-memops -prefetch-hints-file=%S/insert-prefetch-invalid-instr.afdo | FileCheck %s
; ModuleID = 'prefetch.cc'
source_filename = "prefetch.cc"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: norecurse nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #0 !dbg !7 {
entry:
  tail call void @llvm.prefetch(ptr inttoptr (i64 291 to ptr), i32 0, i32 0, i32 1), !dbg !9
  ret i32 291, !dbg !11
}

; Function Attrs: inaccessiblemem_or_argmemonly nounwind
declare void @llvm.prefetch(ptr nocapture readonly, i32, i32, i32) #1

attributes #0 = {"target-cpu"="x86-64" "target-features"="+sse4.2,+ssse3"}
attributes #1 = { inaccessiblemem_or_argmemonly nounwind }
attributes #2 = { argmemonly nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, debugInfoForProfiling: true)
!1 = !DIFile(filename: "prefetch.cc", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 7.0.0 (trunk 327078) (llvm/trunk 327086)"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 8, type: !8, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 12, column: 3, scope: !7)
!10 = !DILocation(line: 14, column: 3, scope: !7)
!11 = !DILocation(line: 15, column: 3, scope: !7)

;CHECK-LABEL: main:
;CHECK:       # %bb.0:
;CHECK:       prefetchnta 291
;CHECK-NOT:   prefetchnta 42(%rax,%ymm0)
