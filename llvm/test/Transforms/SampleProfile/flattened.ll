; Check flattened profile will not be read in thinlto postlink.
; RUN: opt < %s -O2 -flattened-profile-used -pgo-kind=pgo-sample-use-pipeline -profile-file=%S/Inputs/flattened.prof -enable-chr=false -perform-thinlto=true -S | FileCheck %s
; RUN: opt < %s -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%S/Inputs/flattened.prof -flattened-profile-used -S | FileCheck %s
;
; Check flattened profile will be read in thinlto prelink.
; RUN: opt < %s -O2 -flattened-profile-used -pgo-kind=pgo-sample-use-pipeline -profile-file=%S/Inputs/flattened.prof -enable-chr=false -prepare-for-thinlto=true -S | FileCheck %s --check-prefix=PRELINK
; RUN: opt < %s -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%S/Inputs/flattened.prof -flattened-profile-used -S | FileCheck %s --check-prefix=PRELINK
;
; Check flattened profile will be read in non-thinlto mode.
; RUN: opt < %s -O2 -flattened-profile-used -pgo-kind=pgo-sample-use-pipeline -profile-file=%S/Inputs/flattened.prof -enable-chr=false -S | FileCheck %s --check-prefix=NOTHINLTO
; RUN: opt < %s -passes='default<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%S/Inputs/flattened.prof -flattened-profile-used -S | FileCheck %s --check-prefix=NOTHINLTO
;
; CHECK-NOT: !{!"ProfileFormat", !"SampleProfile"}
; PRELINK:   !{!"ProfileFormat", !"SampleProfile"}
; NOTHINLTO: !{!"ProfileFormat", !"SampleProfile"}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local i32 @foo() local_unnamed_addr !dbg !7 {
entry:
  ret i32 -1, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.0 (trunk 345241)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "a.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (trunk 345241)"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 2, column: 3, scope: !7)
