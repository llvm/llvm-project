;; Test that duplicate (identical) MemProf function clone aliases to a CFI jump
;; table canonical function are preserved by LowerTypeTests in the ThinLTO
;; backend (aliasing the renamed .cfi definition), rather than being renamed to
;; .memprof.N.cfi and replaced with undefined declarations expecting jump table
;; entries in the merged module (which only exist for aliases known at the thin
;; link), and without duplicating the function body.
;;
;; Tests three CFI canonical function cases:
;; 1. External function (_Z3foov in cfiFunctionDefs) with a pre-existing source
;;    alias (_Z8fooAliasv, which is still redirected to its jump table entry).
;; 2. Address-taken internal function exported via a split-LTO-unit alias
;;    (_Z8fooAliasv in cfiFunctionDefs) that was also promoted by the thin link.
;; 3. Address-taken internal function (local_foo) exported via a split-LTO-unit
;;    alias (local_foo.1234 in cfiFunctionDefs) that remains internal.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llvm-as src.ll -o src.o
; RUN: llvm-as src.o.thinlto.ll -o src.o.thinlto.bc
; RUN: opt -passes=assign-guid,memprof-context-disambiguation,lowertypetests -stats \
; RUN:  -memprof-import-summary=src.o.thinlto.bc \
; RUN:  -lowertypetests-summary-action=import \
; RUN:  -lowertypetests-read-summary=src.o.thinlto.ll \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  src.o -S 2>&1 | FileCheck %s --check-prefixes=CHECK,EXT \
; RUN:  --implicit-check-not=".memprof.{{[0-9]+}}.cfi"

; RUN: llvm-as src.o.thinlto.promoted.ll -o src.o.thinlto.promoted.bc
; RUN: opt -passes=assign-guid,memprof-context-disambiguation,lowertypetests -stats \
; RUN:  -memprof-import-summary=src.o.thinlto.promoted.bc \
; RUN:  -lowertypetests-summary-action=import \
; RUN:  -lowertypetests-read-summary=src.o.thinlto.promoted.ll \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  src.o -S 2>&1 | FileCheck %s --check-prefixes=CHECK,PROMOTED \
; RUN:  --implicit-check-not=".memprof.{{[0-9]+}}.cfi"

;; Duplicate clones remain aliases (no extra function bodies cloned).
; CHECK: created clone bar.memprof.1
; CHECK: created clone decl bar.memprof.2
; CHECK: created clone decl bar.memprof.3
; CHECK: created clone _Z3foov.memprof.1
; CHECK: created clone alias _Z3foov.memprof.2
; CHECK: created clone alias _Z3foov.memprof.3
; CHECK: created clone local_foo.memprof.1
; CHECK: created clone alias local_foo.memprof.2

;; Case 1 (EXT): _Z3foov is in cfiFunctionDefs. Pre-existing source alias
;; _Z8fooAliasv gets .cfi + a jump-table declaration, while MemProf duplicate
;; aliases _Z3foov.memprof.2 and _Z8fooAliasv.memprof.2 remain defined as
;; aliases to @_Z3foov.cfi.
; EXT-DAG: @_Z8fooAliasv.cfi = alias ptr (...), ptr @_Z3foov.cfi{{$}}
; EXT-DAG: @_Z8fooAliasv.memprof.1 = alias ptr (...), ptr @_Z3foov.memprof.1{{$}}
; EXT-DAG: @_Z3foov.memprof.2 = alias ptr (), ptr @_Z3foov.cfi{{$}}
; EXT-DAG: @_Z8fooAliasv.memprof.2 = alias ptr (...), ptr @_Z3foov.cfi{{$}}
; EXT-DAG: @_Z3foov.memprof.3 = alias ptr (), ptr @_Z3foov.memprof.1{{$}}
; EXT-DAG: @_Z8fooAliasv.memprof.3 = alias ptr (...), ptr @_Z3foov.memprof.1{{$}}

;; Case 2 (PROMOTED): _Z8fooAliasv is in cfiFunctionDefs (collapsed into
;; _Z3foov, which was promoted by ThinLink). MemProf duplicate aliases remain
;; defined as aliases to @_Z8fooAliasv.cfi, and the promoted name _Z3foov is
;; recreated as an alias to @_Z8fooAliasv.cfi.
; PROMOTED-DAG: @_Z8fooAliasv.memprof.1 = alias ptr (...), ptr @_Z3foov.memprof.1{{$}}
; PROMOTED-DAG: @_Z3foov.memprof.2 = alias ptr (), ptr @_Z8fooAliasv.cfi{{$}}
; PROMOTED-DAG: @_Z8fooAliasv.memprof.2 = alias ptr (...), ptr @_Z8fooAliasv.cfi{{$}}
; PROMOTED-DAG: @_Z3foov = alias ptr (), ptr @_Z8fooAliasv.cfi{{$}}

;; Case 3 (both runs): internal local_foo exported via local_foo.1234 in
;; cfiFunctionDefs. Both local_foo.memprof.2 (internal) and
;; local_foo.1234.memprof.2 (hidden) remain defined as aliases to
;; @local_foo.1234.cfi rather than becoming external declarations.
; CHECK-DAG: @local_foo.1234.memprof.1 = hidden alias ptr (), ptr @local_foo.memprof.1{{$}}
; CHECK-DAG: @local_foo.memprof.2 = internal alias ptr (), ptr @local_foo.1234.cfi{{$}}
; CHECK-DAG: @local_foo.1234.memprof.2 = hidden alias ptr (), ptr @local_foo.1234.cfi{{$}}

; EXT-LABEL: define hidden ptr @_Z3foov.cfi()
; PROMOTED-LABEL: define hidden ptr @_Z8fooAliasv.cfi()
; CHECK-LABEL: define hidden ptr @local_foo.1234.cfi()
; CHECK-LABEL: define ptr @_Z3foov.memprof.1()
; CHECK-LABEL: define internal ptr @local_foo.memprof.1()

; EXT-DAG: declare ptr @_Z3foov()
; EXT-DAG: declare ptr @_Z8fooAliasv()
; PROMOTED-DAG: declare ptr @_Z8fooAliasv()
; CHECK-DAG: declare hidden ptr @local_foo.1234()

; CHECK: 5 memprof-context-disambiguation - Number of function clone duplicates detected during ThinLTO backend
; CHECK: 3 memprof-context-disambiguation - Number of function clones created during ThinLTO backend

;--- src.ll
source_filename = "memprof-dups-cfi.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_Z8fooAliasv = alias ptr (...), ptr @_Z3foov
@local_foo.1234 = hidden alias ptr (), ptr @local_foo

define i32 @main() #0 {
entry:
  %call = call ptr @bar(), !callsite !0
  %call1 = call ptr @bar(), !callsite !1
  %call2 = call ptr @local_foo(), !callsite !9
  %call3 = call ptr @local_foo(), !callsite !10
  ret i32 0
}

define available_externally i32 @bar() #0 {
entry:
  %call = call ptr @_Z8fooAliasv(), !callsite !8
  ret i32 0
}

declare ptr @_Znam(i64)

define ptr @_Z3foov() #0 {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !2, !callsite !7
  ret ptr null
}

define internal ptr @local_foo() #0 {
entry:
  %call = call ptr @_Znam(i64 0), !memprof !11, !callsite !16
  ret ptr null
}

attributes #0 = { noinline optnone }

!0 = !{i64 8632435727821051414}
!1 = !{i64 -3421689549917153178}
!2 = !{!3, !5}
!3 = !{!4, !"notcold"}
!4 = !{i64 9086428284934609951, i64 1234, i64 8632435727821051414}
!5 = !{!6, !"cold"}
!6 = !{i64 9086428284934609951, i64 1234, i64 -3421689549917153178}
!7 = !{i64 9086428284934609951}
!8 = !{i64 1234}
!9 = !{i64 111}
!10 = !{i64 222}
!11 = !{!12, !14}
!12 = !{!13, !"notcold"}
!13 = !{i64 333, i64 111}
!14 = !{!15, !"cold"}
!15 = !{i64 333, i64 222}
!16 = !{i64 333}

;--- src.o.thinlto.ll
; ModuleID = 'src.o.thinlto.ll'
source_filename = "src.o.thinlto.bc"

^0 = module: (path: "src.o", hash: (1720506022, 1575514144, 2506794664, 3599359797, 3160884478))
^1 = gv: (guid: 6583049656999245004, summaries: (alias: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), aliasee: ^2)))
^2 = gv: (guid: 9191153033785521275, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 2, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), allocs: ((versions: (notcold, cold, notcold, cold), memProf: ((type: notcold, stackIds: (1234, 8632435727821051414)), (type: cold, stackIds: (1234, 15025054523792398438))))))))
^3 = gv: (guid: 15822663052811949562, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 3, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^4), (callee: ^8)), callsites: ((callee: ^4, clones: (0), stackIds: (8632435727821051414)), (callee: ^4, clones: (1), stackIds: (15025054523792398438)), (callee: ^8, clones: (0), stackIds: (111)), (callee: ^8, clones: (1), stackIds: (222))))))
^4 = gv: (guid: 16434608426314478903, summaries: (function: (module: ^0, flags: (linkage: available_externally, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 2, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^1)), callsites: ((callee: ^1, clones: (0, 1, 0, 1), stackIds: (1234))))))
^8 = gv: (guid: 13482459830562554901, summaries: (function: (module: ^0, flags: (linkage: internal, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 2, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), allocs: ((versions: (notcold, cold, notcold), memProf: ((type: notcold, stackIds: (111)), (type: cold, stackIds: (222))))))))
^5 = cfiFunctionDefs: ((guid: 9191153033785521275, name: "_Z3foov"), (guid: 13529402235550403658, name: "local_foo.1234"))
^6 = flags: 353
^7 = blockcount: 0

;--- src.o.thinlto.promoted.ll
; ModuleID = 'src.o.thinlto.promoted.ll'
source_filename = "src.o.thinlto.promoted.bc"

^0 = module: (path: "src.o", hash: (1720506022, 1575514144, 2506794664, 3599359797, 3160884478))
^1 = gv: (guid: 6583049656999245004, summaries: (alias: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), aliasee: ^2)))
^2 = gv: (guid: 9191153033785521275, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 2, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), allocs: ((versions: (notcold, cold, notcold, cold), memProf: ((type: notcold, stackIds: (1234, 8632435727821051414)), (type: cold, stackIds: (1234, 15025054523792398438))))))))
^3 = gv: (guid: 15822663052811949562, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 3, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^4), (callee: ^8)), callsites: ((callee: ^4, clones: (0), stackIds: (8632435727821051414)), (callee: ^4, clones: (1), stackIds: (15025054523792398438)), (callee: ^8, clones: (0), stackIds: (111)), (callee: ^8, clones: (1), stackIds: (222))))))
^4 = gv: (guid: 16434608426314478903, summaries: (function: (module: ^0, flags: (linkage: available_externally, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 2, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^1)), callsites: ((callee: ^1, clones: (0, 1, 0, 1), stackIds: (1234))))))
^8 = gv: (guid: 13482459830562554901, summaries: (function: (module: ^0, flags: (linkage: internal, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 2, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), allocs: ((versions: (notcold, cold, notcold), memProf: ((type: notcold, stackIds: (111)), (type: cold, stackIds: (222))))))))
^5 = cfiFunctionDefs: ((guid: 6583049656999245004, name: "_Z8fooAliasv"), (guid: 13529402235550403658, name: "local_foo.1234"))
^6 = flags: 353
^7 = blockcount: 0
