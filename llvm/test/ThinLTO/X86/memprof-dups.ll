;; Check that duplicate spurious duplicate (identical) clones are simply
;; created as aliases to the first identical copy, rather than creating
;; multiple clones that call the same callee clones or have the same
;; allocation types. This currently happens in some cases due to additional
;; cloning performed during function assignment.
;;
;; The ThinLTO combined summary was manually modified as described there
;; to force multiple identical copies of various functions.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llvm-as src.ll -o src.o
; RUN: llvm-as src.o.thinlto.ll -o src.o.thinlto.bc
; RUN: opt -passes=memprof-context-disambiguation -stats \
; RUN: 	-memprof-import-summary=src.o.thinlto.bc \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:	src.o -S 2>&1 | FileCheck %s

; CHECK: created clone bar.memprof.1
;; Duplicates of bar are created as declarations since bar is available_externally,
;; and the compiler does not well support available_externally aliases.
; CHECK: created clone decl bar.memprof.2
; CHECK: created clone decl bar.memprof.3
; CHECK: created clone _Z3foov.memprof.1
;; Duplicates of _Z3foov are created as aliases to the appropriate materialized
;; clone of _Z3foov.
; CHECK: created clone alias _Z3foov.memprof.2
; CHECK: created clone alias _Z3foov.memprof.3

;--- src.ll
source_filename = "memprof-distrib-alias.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_Z8fooAliasv = alias ptr (...), ptr @_Z3foov

;; Original alias is unchanged.
; CHECK: @_Z8fooAliasv = alias ptr (...), ptr @_Z3foov{{$}}
;; We create an equivalent alias for the cloned def @_Z3foov.memprof.1.
; CHECK: @_Z8fooAliasv.memprof.1 = alias ptr (...), ptr @_Z3foov.memprof.1

;; We should also create aliases for the duplicate clones of _Z3foov
;; (_Z3foov.memprof.2 and _Z3foov.memprof.3) to the versions they are duplicates
;; of, and ditto for the associated @_Z8fooAliasv clones.
;;
;; _Z3foov.memprof.2 is a duplicate of original _Z3foov, and thus so is _Z8fooAliasv.memprof.2
; CHECK: @_Z3foov.memprof.2 = alias ptr (), ptr @_Z3foov{{$}}
; CHECK: @_Z8fooAliasv.memprof.2 = alias ptr (...), ptr @_Z3foov{{$}}
;; _Z3foov.memprof.3 is a duplicate of _Z3foov.memprof.1, and thus so is _Z8fooAliasv.memprof.3
; CHECK: @_Z3foov.memprof.3 = alias ptr (), ptr @_Z3foov.memprof.1
; CHECK: @_Z8fooAliasv.memprof.3 = alias ptr (...), ptr @_Z3foov.memprof.1

; CHECK-LABEL: define i32 @main()
define i32 @main() #0 {
entry:
  ;; The first call to bar does not allocate cold memory. It should call
  ;; the original function, which eventually calls the original allocation
  ;; decorated with a "notcold" attribute.
  ; CHECK:   call {{.*}} @bar()
  %call = call ptr @bar(), !callsite !0
  ;; The second call to bar allocates cold memory. It should call the cloned
  ;; function which eventually calls a cloned allocation decorated with a
  ;; "cold" attribute.
  ; CHECK:   call {{.*}} @bar.memprof.1()
  %call1 = call ptr @bar(), !callsite !1
  ret i32 0
}

; CHECK-LABEL: define available_externally i32 @bar()
define available_externally i32 @bar() #0 {
entry:
  ; CHECK: call {{.*}} @_Z8fooAliasv()
  %call = call ptr @_Z8fooAliasv(), !callsite !8
  ret i32 0
}

declare ptr @_Znam(i64)

; CHECK-LABEL: define ptr @_Z3foov()
define ptr @_Z3foov() #0 {
entry:
  ; CHECK: call {{.*}} @_Znam(i64 0) #[[NOTCOLD:[0-9]+]]
  %call = call ptr @_Znam(i64 0), !memprof !2, !callsite !7
  ret ptr null
}

; We create actual clone for bar.memprof.1.
; CHECK: define available_externally i32 @bar.memprof.1()
; CHECK:   call {{.*}} @_Z3foov.memprof.1()

;; bar.memprof.2 and bar.memprof.3 are duplicates (of original bar and
;; bar.memprof.1, respectively). However, they are available externally,
;; so rather than create an alias we simply create a declaration, since the
;; compiler does not fully support available_externally aliases.
; CHECK: declare i32 @bar.memprof.2
; CHECK: declare i32 @bar.memprof.3

; We create actual clone for foo.memprof.1.
; CHECK: define {{.*}} @_Z3foov.memprof.1()
; CHECK:   call {{.*}} @_Znam(i64 0) #[[COLD:[0-9]+]]

; CHECK: attributes #[[NOTCOLD]] = { "memprof"="notcold" }
; CHECK: attributes #[[COLD]] = { "memprof"="cold" }

; CHECK: 4 memprof-context-disambiguation - Number of function clone duplicates detected during ThinLTO backend
; CHECK: 2 memprof-context-disambiguation - Number of function clones created during ThinLTO backend

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

;--- src.o.thinlto.ll
; ModuleID = 'src.o.thinlto.ll'
source_filename = "src.o.thinlto.bc"

^0 = module: (path: "src.o", hash: (1720506022, 1575514144, 2506794664, 3599359797, 3160884478))
^1 = gv: (guid: 6583049656999245004, summaries: (alias: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), aliasee: ^2)))
;; Summary for _Z3foov, where the allocs part has been manually modified to add
;; two additional clones that are the same as the prior versions:
;;	... allocs: ((versions: (notcold, cold, notcold, cold), ...
^2 = gv: (guid: 9191153033785521275, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 2, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), allocs: ((versions: (notcold, cold, notcold, cold), memProf: ((type: notcold, stackIds: (1234, 8632435727821051414)), (type: cold, stackIds: (1234, 15025054523792398438))))))))
^3 = gv: (guid: 15822663052811949562, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 3, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^4)), callsites: ((callee: ^4, clones: (0), stackIds: (8632435727821051414)), (callee: ^4, clones: (1), stackIds: (15025054523792398438))))))
;; Summary for bar, where the callsites part has been manually modified to add
;; two additional clones that are the same as the prior clones:
;;	... callsites: ((callee: ^1, clones: (0, 1, 0, 1), ...
^4 = gv: (guid: 16434608426314478903, summaries: (function: (module: ^0, flags: (linkage: available_externally, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 2, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 0, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0), calls: ((callee: ^1)), callsites: ((callee: ^1, clones: (0, 1, 0, 1), stackIds: (1234))))))
^6 = flags: 353
^7 = blockcount: 0
