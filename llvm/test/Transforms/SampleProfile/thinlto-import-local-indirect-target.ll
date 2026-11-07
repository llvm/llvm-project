; ThinLTO pre-link defers indirect call promotion to the backend and records
; import GUIDs for the profiled targets. Normally only targets defined in
; another module get one. A target defined in this module needs one too: when
; another module imports dispatch, it must also import the targets to promote
; them. Check that -sample-profile-preserve-deferred-inlines adds them,
; including nested ones, and that dispatch's summary then has edges to them.
;
; The indirect call at probe 2 has profiled targets hot, local (internal),
; mid (nested in hot), and zero (no samples; in cs.txt also marked to be
; inlined). direct calls the external ext, which is not deferred, so its local
; nested target mid gets no GUID.

; RUN: split-file %s %t
; RUN: opt %t/a.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -profile-summary-hot-count=1000 -sample-profile-file=%t/nested.txt -S \
; RUN:   | FileCheck %s --check-prefixes=CHECK,OFF
; RUN: opt %t/a.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -profile-summary-hot-count=1000 -sample-profile-file=%t/nested.txt -S \
; RUN:   -sample-profile-preserve-deferred-inlines | FileCheck %s --check-prefixes=CHECK,ON
; RUN: opt %t/a.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -profile-summary-hot-count=1000 -sample-profile-file=%t/nested.txt -S \
; RUN:   -sample-profile-preserve-deferred-inlines -sample-profile-prioritized-inline=false \
; RUN:   | FileCheck %s --check-prefixes=CHECK,ON
; RUN: opt %t/a.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -profile-summary-hot-count=1000 -sample-profile-file=%t/cs.txt -S \
; RUN:   -sample-profile-preserve-deferred-inlines | FileCheck %s --check-prefixes=CHECK,ON
; RUN: opt %t/a.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -profile-summary-hot-count=1000 -sample-profile-file=%t/nested.txt \
; RUN:   -sample-profile-preserve-deferred-inlines -thinlto-bc -o - | llvm-dis \
; RUN:   | FileCheck %s --check-prefix=SUMMARY

;; GUIDs: ext -2953423517159131221, hot -8269091651996404185,
;; mid 4395024877212874786, local -8436167952830062855 (from "a.c;local").
; CHECK: define {{.*}} @dispatch({{.*}} !prof ![[#DISPATCH:]]
; CHECK: define {{.*}} @direct({{.*}} !prof ![[#DIRECT:]]
; OFF: ![[#DISPATCH]] = !{!"function_entry_count", i64 1000}
; ON:  ![[#DISPATCH]] = !{!"function_entry_count", i64 1000, i64 4395024877212874786, i64 -8436167952830062855, i64 -8269091651996404185}
; CHECK: ![[#DIRECT]] = !{!"function_entry_count", i64 1000, i64 -2953423517159131221}

; SUMMARY: ^[[#MID:]] = gv: (name: "mid"
; SUMMARY: ^[[#LOCAL:]] = gv: (name: "local"
; SUMMARY: ^[[#HOT:]] = gv: (name: "hot"
; SUMMARY: name: "dispatch", {{.*}}calls: ((callee: ^[[#HOT]], hotness: critical), (callee: ^[[#MID]], hotness: critical), (callee: ^[[#LOCAL]], hotness: critical))

;--- a.ll
source_filename = "a.c"
target triple = "x86_64-unknown-linux-gnu"

@fp = global ptr @local

define i32 @dispatch(ptr %f) #0 !dbg !6 {
  call void @llvm.pseudoprobe(i64 -7762558725055652451, i64 1, i32 0, i64 -1), !dbg !7
  %r = call i32 %f(), !dbg !8
  ret i32 %r
}

define i32 @direct() #0 !dbg !10 {
  call void @llvm.pseudoprobe(i64 -9033704996407825796, i64 1, i32 0, i64 -1), !dbg !11
  %r = call i32 @ext(), !dbg !12
  ret i32 %r
}

define i32 @hot() {
  ret i32 1
}

define internal i32 @local() {
  ret i32 2
}

define i32 @mid() {
  ret i32 3
}

define i32 @zero() {
  ret i32 4
}

declare i32 @ext()
declare void @llvm.pseudoprobe(i64, i64, i32, i64)

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!llvm.pseudo_probe_desc = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "a.c", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i64 -7762558725055652451, i64 1, !"dispatch"}
!4 = !{i64 -9033704996407825796, i64 1, !"direct"}
!5 = !DISubroutineType(types: !{})
!6 = distinct !DISubprogram(name: "dispatch", file: !1, type: !5, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DILocation(line: 1, scope: !6)
!8 = !DILocation(line: 2, scope: !9)
;; Discriminator 119537687 encodes indirect call probe 2.
!9 = !DILexicalBlockFile(scope: !6, file: !1, discriminator: 119537687)
!10 = distinct !DISubprogram(name: "direct", file: !1, type: !5, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DILocation(line: 1, scope: !10)
!12 = !DILocation(line: 2, scope: !13)
;; Discriminator 186646551 encodes direct call probe 2.
!13 = !DILexicalBlockFile(scope: !10, file: !1, discriminator: 186646551)

;--- nested.txt
dispatch:7000:1000
 1: 1000
 2: hot:3000
  1: 2000
  2: mid:1000
   1: 1000
   !Attributes: 3
  !Attributes: 3
 2: local:2000
  1: 2000
  !Attributes: 3
 2: zero:0
  1: 0
  !Attributes: 3
 !CFGChecksum: 1
direct:3000:1000
 1: 1000
 2: ext:2000
  1: 1000
  2: mid:1000
   1: 1000
   !Attributes: 3
  !Attributes: 3
 !CFGChecksum: 1

;--- cs.txt
[dispatch]:1000:1000
 1: 1000
 !CFGChecksum: 1
[dispatch:2 @ hot]:2000:2000
 1: 2000
 !CFGChecksum: 1
 !Attributes: 2
[dispatch:2 @ hot:2 @ mid]:1000:1000
 1: 1000
 !CFGChecksum: 1
 !Attributes: 2
[dispatch:2 @ local]:2000:2000
 1: 2000
 !CFGChecksum: 1
 !Attributes: 2
[dispatch:2 @ zero]:0:0
 1: 0
 !CFGChecksum: 1
 !Attributes: 2
[direct]:1000:1000
 1: 1000
 !CFGChecksum: 1
[direct:2 @ ext]:1000:1000
 1: 1000
 !CFGChecksum: 1
 !Attributes: 2
[direct:2 @ ext:2 @ mid]:1000:1000
 1: 1000
 !CFGChecksum: 1
 !Attributes: 2
