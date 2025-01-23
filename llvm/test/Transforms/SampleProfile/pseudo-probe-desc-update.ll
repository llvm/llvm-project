; RUN: opt < %s -passes='pseudo-probe-desc-update' -S | FileCheck %s

; CHECK:      !llvm.pseudo_probe_desc = !{!0, !1, !2, !3, !4, !5}
; CHECK:      !0 = !{i64 -2012135647395072713, i64 4294967295, !"bar"}
; CHECK-NEXT: !1 = !{i64 9204417991963109735, i64 72617220756, !"work"}
; CHECK-NEXT: !2 = !{i64 6699318081062747564, i64 844700110938769, !"foo"}
; CHECK-NEXT: !3 = !{i64 -2624081020897602054, i64 281563657672557, !"main"}
; CHECK-NEXT: !4 = !{i64 6028998432455395745, i64 0, !"extract1"}
; CHECK-NEXT: !5 = !{i64 -8314581669044049530, i64 0, !"_Zextract2v"}

target triple = "x86_64-unknown-linux-gnu"

define void @extract1() !dbg !1 {
entry:
  ret void
}

define void @extract2() !dbg !2 {
entry:
  ret void
}

!llvm.pseudo_probe_desc = !{!4, !5, !6, !7}
!llvm.module.flags = !{!8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, emissionKind: FullDebug,  splitDebugInlining: false, debugInfoForProfiling: true)
!1 = distinct !DISubprogram(name: "extract1", unit: !0)
!2 = distinct !DISubprogram(name: "extract2", linkageName: "_Zextract2v", unit: !0)
!3 = !DIFile(filename: "foo.c", directory: "/home/test")
!4 = !{i64 -2012135647395072713, i64 4294967295, !"bar"}
!5 = !{i64 9204417991963109735, i64 72617220756, !"work"}
!6 = !{i64 6699318081062747564, i64 844700110938769, !"foo"}
!7 = !{i64 -2624081020897602054, i64 281563657672557, !"main"}
!8 = !{i32 7, !"Dwarf Version", i32 5}
!9 = !{i32 2, !"Debug Info Version", i32 3}
