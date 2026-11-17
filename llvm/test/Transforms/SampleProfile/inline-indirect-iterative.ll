; RUN: split-file %s %t
; RUN: cat %t/call.ll %t/common.ll > %t/input.ll
; RUN: cat %t/invoke.ll %t/common.ll > %t/invoke-input.ll
; RUN: sed "s/alwaysinline/noinline/" %t/input.ll > %t/noinline-input.ll
; RUN: opt -passes=sample-profile -sample-profile-file=%S/Inputs/inline-indirect-iterative.prof -sample-profile-prioritized-inline=false -sample-profile-merge-inlinee=true -profile-summary-hot-count=100  -S %t/input.ll | FileCheck %s --check-prefixes=CHECK,INLINED,DIRECT
; RUN: opt -passes=sample-profile -sample-profile-file=%S/Inputs/inline-indirect-iterative.prof -sample-profile-prioritized-inline=false -sample-profile-merge-inlinee=false -profile-summary-hot-count=100  -S %t/input.ll | FileCheck %s --check-prefixes=CHECK,INLINED,DIRECT
; RUN: opt -passes=sample-profile -sample-profile-file=%S/Inputs/inline-indirect-iterative.prof -sample-profile-prioritized-inline=false -sample-profile-merge-inlinee=true -profile-summary-hot-count=100 -sample-profile-icp-max-prom=1 -S %t/input.ll | FileCheck %s --check-prefixes=CHECK,INLINED
; RUN: opt -passes=sample-profile -sample-profile-file=%S/Inputs/inline-indirect-iterative.prof -sample-profile-prioritized-inline=false -sample-profile-merge-inlinee=true -profile-summary-hot-count=100 -sample-profile-icp-max-prom=0 -S %t/input.ll | FileCheck %s --check-prefixes=CHECK,OUTLINED,NOPROM
; RUN: opt -passes=sample-profile -sample-profile-file=%S/Inputs/inline-indirect-iterative.prof -sample-profile-prioritized-inline=false -sample-profile-merge-inlinee=true -profile-summary-hot-count=100 --disable-sample-loader-inlining -S %t/input.ll | FileCheck %s --check-prefixes=CHECK,OUTLINED,NOPROM
; RUN: opt -passes=sample-profile -sample-profile-file=%S/Inputs/inline-indirect-iterative.prof -sample-profile-prioritized-inline=false -sample-profile-merge-inlinee=true -profile-summary-hot-count=100  -S %t/noinline-input.ll | FileCheck %s --check-prefixes=CHECK,OUTLINED,DIRECT
; RUN: opt -passes=sample-profile -sample-profile-file=%S/Inputs/inline-indirect-iterative.prof -sample-profile-prioritized-inline=false -sample-profile-merge-inlinee=true -profile-summary-hot-count=100  -S %t/invoke-input.ll | FileCheck %s --check-prefixes=CHECK,INLINED,DIRECT
; RUN: opt -passes=sample-profile -sample-profile-file=%S/Inputs/inline-indirect-iterative.prof -sample-profile-prioritized-inline=false -sample-profile-merge-inlinee=true -profile-summary-hot-count=100 -sample-profile-icp-max-prom=1 -S %t/input.ll -o %t/once.ll
; RUN: opt -passes=sample-profile -sample-profile-file=%S/Inputs/inline-indirect-iterative.prof -sample-profile-prioritized-inline=false -sample-profile-merge-inlinee=true -profile-summary-hot-count=100 -sample-profile-icp-max-prom=0 -S %t/once.ll | FileCheck %s --check-prefixes=CHECK,INLINED
; RUN: llvm-profdata merge --sample --extbinary --use-md5 %S/Inputs/inline-indirect-iterative.prof -o %t/md5.prof
; RUN: opt -passes=sample-profile -sample-profile-file=%t/md5.prof -sample-profile-prioritized-inline=false -sample-profile-merge-inlinee=true -profile-summary-hot-count=100  -S %t/input.ll | FileCheck %s --check-prefixes=CHECK,INLINED,DIRECT
;
; Scan surviving calls only after iterative inlining finishes. A promoted and
; inlined target must not be merged from the fallback on the next iteration.
; Recover both unpromoted targets and promoted direct calls that did not inline,
; including when no successful inline triggers another iteration. Read existing
; promotion history even when a later invocation lowers the promotion limit.
;
; CHECK-LABEL: define i32 @dispatch(
; NOPROM-NOT: icmp eq ptr
; INLINED: icmp eq ptr %fp, @hot
; INLINED: add i32 %x, 1
; DIRECT: {{call|invoke}} i32 @warm(i32 %x)
; CHECK: {{call|invoke}} i32 %fp(i32 %x)
; CHECK-LABEL: define i32 @hot(
; CHECK-SAME: !prof ![[HOT:[0-9]+]]
; CHECK-LABEL: define i32 @warm(
; CHECK-SAME: !prof ![[WARM:[0-9]+]]
; INLINED-DAG: ![[HOT]] = !{!"function_entry_count", i64 11}
; OUTLINED-DAG: ![[HOT]] = !{!"function_entry_count", i64 1011}
; CHECK-DAG: ![[WARM]] = !{!"function_entry_count", i64 521}
;
;--- call.ll
target triple = "x86_64-unknown-linux-gnu"

define i32 @dispatch(ptr %fp, i32 %x) #0 !dbg !10 {
  %r = call i32 %fp(i32 %x), !dbg !20
  ret i32 %r, !dbg !20
}

;--- invoke.ll
target triple = "aarch64-unknown-linux-gnu"
declare i32 @__gxx_personality_v0(...)

define i32 @dispatch(ptr %fp, i32 %x) #0 personality ptr @__gxx_personality_v0 !dbg !10 {
  %r = invoke i32 %fp(i32 %x) to label %exit unwind label %unwind, !dbg !20
exit:
  ret i32 %r, !dbg !20
unwind:
  %lp = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %lp
}

;--- common.ll
define i32 @hot(i32 %x) #1 !dbg !11 {
  %r = add i32 %x, 1, !dbg !21
  ret i32 %r, !dbg !21
}

define i32 @warm(i32 %x) #2 !dbg !12 {
  %r = add i32 %x, 2, !dbg !22
  ret i32 %r, !dbg !22
}

attributes #0 = { "use-sample-profile" }
attributes #1 = { alwaysinline "use-sample-profile" }
attributes #2 = { noinline "use-sample-profile" }
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "repro", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "repro.cpp", directory: "/")
!2 = !DISubroutineType(types: !4)
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{}
!10 = distinct !DISubprogram(name: "dispatch", scope: !1, file: !1, line: 10, type: !2, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = distinct !DISubprogram(name: "hot", scope: !1, file: !1, line: 20, type: !2, scopeLine: 20, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = distinct !DISubprogram(name: "warm", scope: !1, file: !1, line: 30, type: !2, scopeLine: 30, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!20 = !DILocation(line: 11, column: 3, scope: !10)
!21 = !DILocation(line: 21, column: 3, scope: !11)
!22 = !DILocation(line: 31, column: 3, scope: !12)
