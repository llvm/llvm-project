; Test summary parsing of index-based WPD related summary fields
; RUN: llvm-as %s -o - | llvm-dis -o %t.ll
; RUN: grep "^\^" %s >%t2
; RUN: grep "^\^" %t.ll >%t3
; Expect that the summary information is the same after round-trip through
; llvm-as and llvm-dis.
; RUN: diff -b %t2 %t3

source_filename = "thinlto-vtable-summary.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.A = type { i32 (...)** }
%struct.B = type { %struct.A }
%struct.C = type { %struct.A }

@_ZTV1B = constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* undef, i8* bitcast (i32 (%struct.B*, i32)* @_ZN1B1fEi to i8*), i8* bitcast (i32 (%struct.A*, i32)* @_ZN1A1nEi to i8*)] }, !type !0, !type !1
@_ZTV1C = constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* undef, i8* bitcast (i32 (%struct.C*, i32)* @_ZN1C1fEi to i8*), i8* bitcast (i32 (%struct.A*, i32)* @_ZN1A1nEi to i8*)] }, !type !0, !type !2

declare i32 @_ZN1B1fEi(%struct.B*, i32)

declare i32 @_ZN1A1nEi(%struct.A*, i32)

declare i32 @_ZN1C1fEi(%struct.C*, i32)

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
!2 = !{i64 16, !"_ZTS1C"}

^0 = module: (path: "<stdin>", hash: (0, 0, 0, 0, 0))
^1 = gv: (name: "_ZN1A1nEi") ; guid = 1621563287929432257
^2 = gv: (name: "_ZTV1B", summaries: (variable: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), varFlags: (readonly: 0, writeonly: 0, constant: 0, vcall_visibility: 0), vTableFuncs: ((virtFunc: ^3, offset: 16), (virtFunc: ^1, offset: 24)), refs: (^3, ^1)))) ; guid = 5283576821522790367
^3 = gv: (name: "_ZN1B1fEi") ; guid = 7162046368816414394
^4 = gv: (name: "_ZTV1C", summaries: (variable: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), varFlags: (readonly: 0, writeonly: 0, constant: 0, vcall_visibility: 0), vTableFuncs: ((virtFunc: ^5, offset: 16), (virtFunc: ^1, offset: 24)), refs: (^1, ^5)))) ; guid = 13624023785555846296
^5 = gv: (name: "_ZN1C1fEi") ; guid = 14876272565662207556
^6 = typeidCompatibleVTable: (name: "_ZTS1A", summary: ((offset: 16, ^2), (offset: 16, ^4))) ; guid = 7004155349499253778
^7 = typeidCompatibleVTable: (name: "_ZTS1B", summary: ((offset: 16, ^2))) ; guid = 6203814149063363976
^8 = typeidCompatibleVTable: (name: "_ZTS1C", summary: ((offset: 16, ^4))) ; guid = 1884921850105019584
^9 = blockcount: 0
