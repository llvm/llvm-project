; RUN: rm -rf %t
; RUN: split-file %s %t
; RUN: llvm-ctxprof-util fromYAML --input=%t/profile.yaml --output=%t/profile.ctxprofdata

; RUN: opt -passes='module-inline,print<ctx-prof-analysis>' -ctx-profile-printer-level=everything %t/module.ll -S \
; RUN:   -use-ctx-profile=%t/profile.ctxprofdata -ctx-profile-printer-level=json \
; RUN:   -o - 2> %t/profile-final.txt | FileCheck %s
; RUN: %python %S/json_equals.py %t/profile-final.txt %t/expected.json

; There are 2 calls to @a from @entrypoint. We only inline the one callsite
; marked as alwaysinline, the rest are blocked (marked noinline). After the inline,
; the updated contextual profile should still have the same tree for the non-inlined case.
; For the inlined case, we should observe, for the @entrypoint context:
;  - an empty callsite where the inlined one was (first one, i.e. 0)
;  - more counters appended to the old counter list (because we ingested the
;    ones from @a). The values are copied.
;  - a new callsite to @b
; CHECK-LABEL: @entrypoint
; CHECK-LABEL: yes:
; CHECK:         call void @llvm.instrprof.increment(ptr @entrypoint, i64 0, i32 3, i32 1)
; CHECK-NEXT:    br label %loop.i
; CHECK-LABEL:  loop.i:
; CHECK-NEXT:    %indvar.i = phi i32 [ %indvar.next.i, %loop.i ], [ 0, %yes ]
; CHECK-NEXT:    call void @llvm.instrprof.increment(ptr @entrypoint, i64 0, i32 2, i32 3)
; CHECK-NEXT:    %b.i = add i32 %x, %indvar.i
; CHECK-NEXT:    call void @llvm.instrprof.callsite(ptr @entrypoint, i64 0, i32 1, i32 2, ptr @b)
; CHECK-NEXT:    %call3.i = call i32 @b() #1
; CHECK-LABEL: no:
; CHECK-NEXT:    call void @llvm.instrprof.increment(ptr @entrypoint, i64 0, i32 3, i32 2)
; CHECK-NEXT:    call void @llvm.instrprof.callsite(ptr @entrypoint, i64 0, i32 2, i32 1, ptr @a)
; CHECK-NEXT:    %call2 = call i32 @a(i32 %x) #1
; CHECK-NEXT:    br label %exit

; Make sure the postlink thinlto pipeline is aware of ctxprof
; RUN: opt -passes='thinlto<O2>' -use-ctx-profile=%t/profile.ctxprofdata \
; RUN:   %t/module.ll -S -o - | FileCheck %s --check-prefix=PIPELINE

; PIPELINE-LABEL: define i32 @entrypoint
; PIPELINE-SAME: !prof ![[ENTRYPOINT_COUNT:[0-9]+]]
; PIPELINE-LABEL: loop.i:
; PIPELINE:         br i1 %cond.i, label %loop.i, label %exit, !prof ![[LOOP_BW_INL:[0-9]+]]
; PIPELINE-LABEL: define i32 @a
; PIPELINE-LABEL: loop:
; PIPELINE:         br i1 %cond, label %loop, label %exit, !prof ![[LOOP_BW_ORIG:[0-9]+]]

; PIPELINE: ![[ENTRYPOINT_COUNT]] = !{!"function_entry_count", i64 10}
; These are the weights of the inlined @a, where the counters were 2, 100 (2 for entry, 100 for loop)
; PIPELINE: ![[LOOP_BW_INL]] = !{!"branch_weights", i32 98, i32 2}
; These are the weights of the un-inlined @a, where the counters were 8, 500 (8 for entry, 500 for loop)
; PIPELINE: ![[LOOP_BW_ORIG]] = !{!"branch_weights", i32 492, i32 8}

;--- module.ll
define i32 @entrypoint(i32 %x) !guid !0 {
  call void @llvm.instrprof.increment(ptr @entrypoint, i64 0, i32 3, i32 0)
  %t = icmp eq i32 %x, 0
  br i1 %t, label %yes, label %no
yes:
  call void @llvm.instrprof.increment(ptr @entrypoint, i64 0, i32 3, i32 1)
  call void @llvm.instrprof.callsite(ptr @entrypoint, i64 0, i32 2, i32 0, ptr @a)
  %call1 = call i32 @a(i32 %x) alwaysinline
  br label %exit
no:
  call void @llvm.instrprof.increment(ptr @entrypoint, i64 0, i32 3, i32 2)
  call void @llvm.instrprof.callsite(ptr @entrypoint, i64 0, i32 2, i32 1, ptr @a)
  %call2 = call i32 @a(i32 %x) noinline
  br label %exit
exit:
  %ret = phi i32 [%call1, %yes], [%call2, %no]
  ret i32 %ret
}

define i32 @a(i32 %x) !guid !1 {
entry:
  call void @llvm.instrprof.increment(ptr @a, i64 0, i32 2, i32 0)
  br label %loop
loop:
  %indvar = phi i32 [%indvar.next, %loop], [0, %entry]
  call void @llvm.instrprof.increment(ptr @a, i64 0, i32 2, i32 1)
  %b = add i32 %x, %indvar
  call void @llvm.instrprof.callsite(ptr @a, i64 0, i32 1, i32 0, ptr @b)
  %call3 = call i32 @b() noinline
  %indvar.next = add i32 %indvar, %call3
  %cond = icmp slt i32 %indvar.next, %x
  br i1 %cond, label %loop, label %exit
exit:
  ret i32 8
}

define i32 @b() !guid !2 {
  call void @llvm.instrprof.increment(ptr @b, i64 0, i32 1, i32 0)
  ret i32 1
}

!0 = !{i64 1000}
!1 = !{i64 1001}
!2 = !{i64 1002}
;--- profile.yaml
- Guid: 1000
  Counters: [10, 2, 8]
  Callsites:  -
                - Guid: 1001
                  Counters: [2, 100]
                  Callsites:  -
                                - Guid: 1002
                                  Counters: [100]
              -
                - Guid: 1001
                  Counters: [8, 500]
                  Callsites:  -
                                - Guid: 1002
                                  Counters: [500]
;--- expected.json
[
  { "Guid": 1000,
    "Counters": [10, 2, 8, 100],
    "Callsites": [
      [],
      [ { "Guid": 1001,
          "Counters": [8, 500],
          "Callsites": [[{"Guid": 1002, "Counters": [500]}]]}
      ],
      [{ "Guid": 1002, "Counters": [100]}]
    ]
  }
]
