; RUN: opt %s -passes='always-inline' -S | FileCheck %s

;; Test that intermediate (TileIR) debug locations are correctly preserved and
;; updated when a function is inlined. The inliner must:
;; 1. Preserve the MDTuple structure {DILocation, {kind, DILocation}} on
;;    inlined instructions.
;; 2. Append an inlinedAt chain to the primary source DILocation.
;; 3. Append a separate inlinedAt chain to the intermediate DILocation.

define void @caller(ptr addrspace(1) %a, ptr addrspace(1) %b) !dbg !20 {
entry:
  call void @callee(ptr addrspace(1) %a, ptr addrspace(1) %b), !dbg !30
  call void @callee_nodebug(ptr addrspace(1) %a, ptr addrspace(1) %b), !dbg !30
  ret void, !dbg !34
}

define void @callee(ptr addrspace(1) %x, ptr addrspace(1) %y) #0 !dbg !5 {
entry:
  %v = load i32, ptr addrspace(1) %x, !dbg !8
  store i32 %v, ptr addrspace(1) %y, !dbg !14
  ret void, !dbg !18
}

;; Nodebug callee (no per-instruction !dbg, no !dbg on the subprogram). Every
;; inlined instruction takes the `I.setDebugLoc(TheCallDL)` fallback in
;; InlineFunction.cpp; that path must preserve the call site's
;; intermediate-loc MDTuple wrapper on TheCallDL.
define void @callee_nodebug(ptr addrspace(1) %x, ptr addrspace(1) %y) #0 {
entry:
  %v = load i32, ptr addrspace(1) %x
  store i32 %v, ptr addrspace(1) %y
  ret void
}

attributes #0 = { alwaysinline }

;; After inlining @callee into @caller, the instructions from @callee should
;; have their source DILocations with an inlinedAt pointing to the call site
;; in @caller. The intermediate locations should also gain an inlinedAt chain.

; CHECK-LABEL: define void @caller
;; First call: regular @callee (has per-instruction !dbg) — exercises the
;; `inlineDebugLoc` path.
; CHECK:      load i32, ptr addrspace(1) %a,{{.*}} !dbg ![[IDBG1:[0-9]+]]
; CHECK-NEXT: store i32 %v.i, ptr addrspace(1) %b,{{.*}} !dbg ![[IDBG2:[0-9]+]]
;; Second call: @callee_nodebug — exercises the `I.setDebugLoc(TheCallDL)`
;; fallback. Both inlined instructions share the SAME !dbg (the call site's
;; TheCallDL, propagated verbatim).
; CHECK-NEXT: load i32, ptr addrspace(1) %a,{{.*}} !dbg ![[NODBG:[0-9]+]]
; CHECK-NEXT: store i32 %v.i{{[0-9]+}}, ptr addrspace(1) %b,{{.*}} !dbg ![[NODBG]]

;; Inlined instructions should have MDTuple debug locations.
; CHECK-DAG: ![[IDBG1]] = !{![[ISRC1:[0-9]+]], ![[IINT1:[0-9]+]]}
; CHECK-DAG: ![[IDBG2]] = !{![[ISRC2:[0-9]+]], ![[IINT2:[0-9]+]]}

;; Source locations should have inlinedAt pointing to call site.
; CHECK-DAG: ![[ISRC1]] = !DILocation(line: 10, column: 1, scope: ![[CALLEE_SP:[0-9]+]], inlinedAt: ![[IA:[0-9]+]])
; CHECK-DAG: ![[ISRC2]] = !DILocation(line: 11, column: 1, scope: ![[CALLEE_SP]], inlinedAt: ![[IA]])

;; Intermediate tuples should contain kind string and DILocation with inlinedAt.
; CHECK-DAG: ![[IINT1]] = !{!"tile ir", ![[IILOC1:[0-9]+]]}
; CHECK-DAG: ![[IINT2]] = !{!"tile ir", ![[IILOC2:[0-9]+]]}
; CHECK-DAG: ![[IILOC1]] = !DILocation(line: 100, column: 1, scope: ![[ISCOPE:[0-9]+]], inlinedAt: ![[INTIA:[0-9]+]])
; CHECK-DAG: ![[IILOC2]] = !DILocation(line: 101, column: 1, scope: ![[ISCOPE]], inlinedAt: ![[INTIA]])

;; The inlinedAt for source locs should reference the caller's call site.
; CHECK-DAG: ![[IA]] = distinct !DILocation(line: 50, column: 1, scope: ![[CALLER_SP:[0-9]+]])

;; The inlinedAt for intermediate locs should reference the caller's intermediate call site.
; CHECK-DAG: ![[INTIA]] = distinct !DILocation(line: 200, column: 1, scope:

;; The shared !dbg on the nodebug-callee inlined instructions must be an
;; MDTuple — primary equals the call site's primary, secondary equals the
;; call site's intermediate. Both are propagated verbatim (no inlined-at
;; chain rewrite for the fallback).
; CHECK-DAG: ![[NODBG]] = !{![[NDSRC:[0-9]+]], ![[NDINT:[0-9]+]]}
; CHECK-DAG: ![[NDSRC]] = !DILocation(line: 50, column: 1, scope:
; CHECK-DAG: ![[NDINT]] = !{!"tile ir", ![[NDIILOC:[0-9]+]]}
; CHECK-DAG: ![[NDIILOC]] = !DILocation(line: 200, column: 1, scope:

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!3}
;; Required by the IR verifier whenever an instruction carries an
;; intermediate-loc !dbg chain: every intermediate DIFile referenced by a
;; chain must be declared here.
!llvm.intermediate_level_source = !{!100, !101}
!100 = !{!"tile ir", !50, !""}
!101 = !{!"tile ir", !80, !""}

!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2, producer: "test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!2 = !DIFile(filename: "test.py", directory: "/test")
!3 = !{i32 2, !"Debug Info Version", i32 3}

; Callee subprogram and locations
!5 = distinct !DISubprogram(name: "callee", scope: !2, file: !2, line: 9, type: !6, spFlags: DISPFlagDefinition, unit: !1)
!6 = !DISubroutineType(types: !7)
!7 = !{null}

;; Callee source locations
!40 = !DILocation(line: 10, column: 1, scope: !5)
!41 = !DILocation(line: 11, column: 1, scope: !5)

;; Callee intermediate file/scope/locations
!50 = !DIFile(filename: "callee.tileir", directory: ".")
!51 = distinct !DILexicalBlockFile(scope: !5, file: !50, discriminator: 0)
!52 = !DILocation(line: 100, column: 1, scope: !51)
!53 = !DILocation(line: 101, column: 1, scope: !51)

;; Callee intermediate tuples
!60 = !{!"tile ir", !52}
!61 = !{!"tile ir", !53}

;; Callee combined !dbg metadata
!8  = !{!40, !60}
!14 = !{!41, !61}
!18 = !DILocation(line: 12, column: 1, scope: !5)

;; Caller subprogram
!20 = distinct !DISubprogram(name: "caller", scope: !2, file: !2, line: 49, type: !6, spFlags: DISPFlagDefinition, unit: !1)

;; Caller call site source location
!70 = !DILocation(line: 50, column: 1, scope: !20)

;; Caller call site intermediate location
!80 = !DIFile(filename: "caller.tileir", directory: ".")
!81 = distinct !DILexicalBlockFile(scope: !20, file: !80, discriminator: 0)
!82 = !DILocation(line: 200, column: 1, scope: !81)
!83 = !{!"tile ir", !82}

;; Caller combined call site !dbg
!30 = !{!70, !83}

;; Caller return (plain, no intermediate)
!34 = !DILocation(line: 51, column: 1, scope: !20)
