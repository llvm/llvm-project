; REQUIRES: x86_64-linux
;
; RUN: rm -rf %t
; RUN: split-file %s %t
; RUN: llvm-ctxprof-util fromYAML --input=%t/profile.yaml --output=%t/profile.ctxprofdata
; RUN: opt -module-summary -passes='thinlto-pre-link<O2>' -use-ctx-profile=%t/profile.ctxprofdata \
; RUN:   %t/example.ll -S -o %t/4909520559318251808.ll
; RUN: FileCheck --input-file %t/4909520559318251808.ll %s --check-prefix=PRELINK

; RUN: opt -passes='ctx-prof-flatten' -use-ctx-profile=%t/profile.ctxprofdata %t/4909520559318251808.ll -S  | FileCheck %s --check-prefix=POSTLINK
;
;
; Check that instrumentation occurs where expected: the "no" block for both foo and
; @an_entrypoint - which explains the subsequent branch weights
;
; PRELINK-LABEL: @foo
; PRELINK-LABEL: yes:
; PRELINK-LABEL: no:
; PRELINK-NEXT:     call void @llvm.instrprof.increment(ptr @foo, i64 [[#]], i32 2, i32 1)

; PRELINK-LABEL: @an_entrypoint
; PRELINK:       br i1 %t, label %yes, label %common.ret, !prof ![[PREPROF:[0-9]+]]
; PRELINK-LABEL: yes:
; PRELINK-NEXT:    call void @llvm.instrprof.increment(ptr @an_entrypoint, i64 [[#]], i32 2, i32 1)
; PRELINK:      ![[#]] = !{i32 1, !"ProfileSummary", !1}
; PRELINK:      ![[#]] = !{!"TotalCount", i64 3595}
; PRELINK:      ![[#]] = !{!"MaxCount", i64 3000}
; PRELINK:      ![[#]] = !{!"MaxInternalCount", i64 3000}
; PRELINK:      ![[#]] = !{!"MaxFunctionCount", i64 300}
; PRELINK:      ![[#]] = !{!"NumCounts", i64 6}
; PRELINK:      ![[#]] = !{!"NumFunctions", i64 3}
; PRELINK:       ![[PREPROF]] = !{!"branch_weights", i32 40, i32 60}

; Check that the output has:
;  - no instrumentation
;  - the 2 functions have an entry count
;  - each conditional branch has profile annotation
;
; POSTLINK-NOT:   call void @llvm.instrprof
;
; make sure we have function entry counts, branch weights, and a profile summary.
; POSTLINK-LABEL: @foo
; POSTLINK-SAME:    !prof ![[FOO_EP:[0-9]+]]
; POSTLINK:          br i1 %t, label %yes, label %no, !prof ![[FOO_BW:[0-9]+]]
; POSTLINK-LABEL: @an_entrypoint
; POSTLINK-SAME:    !prof ![[AN_ENTRYPOINT_EP:[0-9]+]]
; POSTLINK:          br i1 %t, label %yes, label %common.ret, !prof ![[AN_ENTRYPOINT_BW:[0-9]+]]

; The postlink summary is restricted to the stuff under the root - including the
; "unhandled" data.
; POSTLINK:      ![[#]] = !{i32 1, !"ProfileSummary", !1}
; POSTLINK:      ![[#]] = !{!"TotalCount", i64 1495}
; POSTLINK:      ![[#]] = !{!"MaxCount", i64 1000}
; POSTLINK:      ![[#]] = !{!"MaxInternalCount", i64 1000}
; POSTLINK:      ![[#]] = !{!"MaxFunctionCount", i64 200}
; POSTLINK:      ![[#]] = !{!"NumCounts", i64 6}
; POSTLINK:      ![[#]] = !{!"NumFunctions", i64 3}

;
; @foo will be called both unconditionally and conditionally, on the "yes" branch
; which has a count of 40. So 140 times.

; POSTLINK:       ![[FOO_EP]] = !{!"function_entry_count", i64 140} 

; foo's "no" branch is taken 10+5 times (from the 2 contexts belonging to foo).
; Which means its "yes" branch is taken 140 - 15 times.

; POSTLINK:       ![[FOO_BW]] = !{!"branch_weights", i32 125, i32 15} 
; POSTLINK:       ![[AN_ENTRYPOINT_EP]] = !{!"function_entry_count", i64 100}
; POSTLINK:       ![[AN_ENTRYPOINT_BW]] = !{!"branch_weights", i32 40, i32 60} 

;--- profile.yaml
Contexts:
  - Guid: 4909520559318251808
    TotalRootEntryCount: 100
    Unhandled:
      - Guid: 1234
        Counters: [200, 1000]
    Counters: [100, 40]
    Callsites: -
                - Guid: 11872291593386833696
                  Counters: [ 100, 5 ]
               -
                - Guid: 11872291593386833696
                  Counters: [ 40, 10 ]
FlatProfiles:
  - Guid: 1234
    Counters: [ 100, 2000 ]
;--- example.ll
declare void @bar()

define void @foo(i32 %a, ptr %fct) #0 !guid !0 {
  %t = icmp sgt i32 %a, 7
  br i1 %t, label %yes, label %no
yes:
  call void %fct(i32 %a)
  br label %exit
no:
  call void @bar()
  br label %exit
exit:
  ret void
}

define void @an_entrypoint(i32 %a) !guid !1 {
  %t = icmp sgt i32 %a, 0
  call void @foo(i32 10, ptr null)
  br i1 %t, label %yes, label %no

yes:
  call void @foo(i32 1, ptr null)
  ret void
no:
  ret void
}

attributes #0 = { noinline }
!0 = !{ i64 11872291593386833696 }
!1 = !{i64 4909520559318251808}
