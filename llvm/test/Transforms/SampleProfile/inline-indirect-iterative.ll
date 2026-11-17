; RUN: opt -passes=sample-profile -sample-profile-file=%S/Inputs/inline-indirect-iterative.prof -sample-profile-prioritized-inline=false -sample-profile-merge-inlinee=true -profile-summary-hot-count=100 -S %s | FileCheck %s
;
; The indirect fallback survives after hot is promoted and inlined. Re-scanning
; it must not merge hot's inline context into the standalone profile.
;
; CHECK-LABEL: define i32 @dispatch(
; CHECK: icmp eq ptr %fp, @hot
; CHECK: add i32 %x, 1
; CHECK: call i32 %fp(i32 %x)
; CHECK-LABEL: define i32 @hot(
; CHECK-SAME: !prof ![[HOT:[0-9]+]]
; CHECK: ![[HOT]] = !{!"function_entry_count", i64 11}
;
target triple = "x86_64-unknown-linux-gnu"

define i32 @dispatch(ptr %fp, i32 %x) #0 !dbg !10 {
  %r = call i32 %fp(i32 %x), !dbg !20
  ret i32 %r, !dbg !20
}

define i32 @hot(i32 %x) #1 !dbg !11 {
  %r = add i32 %x, 1, !dbg !21
  ret i32 %r, !dbg !21
}

attributes #0 = { "use-sample-profile" }
attributes #1 = { alwaysinline "use-sample-profile" }
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "repro", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "repro.cpp", directory: "/")
!2 = !DISubroutineType(types: !4)
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{}
!10 = distinct !DISubprogram(name: "dispatch", scope: !1, file: !1, line: 10, type: !2, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = distinct !DISubprogram(name: "hot", scope: !1, file: !1, line: 20, type: !2, scopeLine: 20, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!20 = !DILocation(line: 11, column: 3, scope: !10)
!21 = !DILocation(line: 21, column: 3, scope: !11)
