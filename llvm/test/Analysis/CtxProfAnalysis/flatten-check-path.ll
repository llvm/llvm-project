; REQUIRES: asserts && x86_64-linux
; Check that the profile annotator works: we hit an exit and non-zero paths to
; already visited blocks count as taken (i.e. the flow continues through them).
;
; RUN: split-file %s %t
; RUN: llvm-ctxprof-util fromYAML --input=%t/profile_ok.yaml --output=%t/profile_ok.ctxprofdata
; RUN: llvm-ctxprof-util fromYAML --input=%t/profile_pump.yaml --output=%t/profile_pump.ctxprofdata
; RUN: llvm-ctxprof-util fromYAML --input=%t/profile_unreachable.yaml --output=%t/profile_unreachable.ctxprofdata
;
; RUN: opt -passes=ctx-prof-flatten %t/example_ok.ll -use-ctx-profile=%t/profile_ok.ctxprofdata -S -o - | FileCheck %s
; RUN: not --crash opt -passes=ctx-prof-flatten %t/message_pump.ll -use-ctx-profile=%t/profile_pump.ctxprofdata -S 2>&1 | FileCheck %s --check-prefix=ASSERTION
; RUN: not --crash opt -passes=ctx-prof-flatten %t/unreachable.ll -use-ctx-profile=%t/profile_unreachable.ctxprofdata -S 2>&1 | FileCheck %s --check-prefix=ASSERTION

; CHECK: br i1 %x, label %b1, label %exit, !prof ![[PROF1:[0-9]+]]
; CHECK: br i1 %y, label %blk, label %exit, !prof ![[PROF2:[0-9]+]]
; CHECK: ![[PROF1]] = !{!"branch_weights", i32 1, i32 1}
; CHECK: ![[PROF2]] = !{!"branch_weights", i32 0, i32 1}
; ASSERTION: Assertion `allTakenPathsExit()

; b1->exit is the only way out from b1, but the exit block would have been
; already visited from blk. That should not result in an assertion, though.
;--- example_ok.ll
define void @foo(i32 %t) !guid !0 {
entry:
  call void @llvm.instrprof.increment(ptr @foo, i64 42, i32 42, i32 0)
  br label %blk
blk:
  call void @llvm.instrprof.increment(ptr @foo, i64 42, i32 42, i32 1)
  %x = icmp eq i32 %t, 0
  br i1 %x, label %b1, label %exit
b1:
  call void @llvm.instrprof.increment(ptr @foo, i64 42, i32 42, i32 2)
  %y = icmp eq i32 %t, 0
  br i1 %y, label %blk, label %exit
exit:
  call void @llvm.instrprof.increment(ptr @foo, i64 42, i32 42, i32 3)
  ret void
}
!0 = !{i64 1234}

;--- profile_ok.yaml
- Guid: 1234 
  Counters: [2, 2, 1, 2]

;--- message_pump.ll
; This is a message pump: the loop never exits. This should result in an
; assertion because we can't reach an exit BB

define void @foo(i32 %t) !guid !0 {
entry:
  call void @llvm.instrprof.increment(ptr @foo, i64 42, i32 42, i32 0)  
  br label %blk
blk:
  call void @llvm.instrprof.increment(ptr @foo, i64 42, i32 42, i32 1)
  %x = icmp eq i32 %t, 0
  br i1 %x, label %blk, label %exit
exit:
  call void @llvm.instrprof.increment(ptr @foo, i64 42, i32 42, i32 2)
  ret void
}
!0 = !{i64 1234}

;--- profile_pump.yaml
- Guid: 1234
  Counters: [2, 10, 0]

;--- unreachable.ll
; An unreachable block is reached, that's an error
define void @foo(i32 %t) !guid !0 {
entry:
  call void @llvm.instrprof.increment(ptr @foo, i64 42, i32 42, i32 0)
  br label %blk
blk:
  call void @llvm.instrprof.increment(ptr @foo, i64 42, i32 42, i32 1)
  %x = icmp eq i32 %t, 0
  br i1 %x, label %b1, label %exit
b1:
  call void @llvm.instrprof.increment(ptr @foo, i64 42, i32 42, i32 2)
  unreachable
exit:
  call void @llvm.instrprof.increment(ptr @foo, i64 42, i32 42, i32 3)
  ret void
}
!0 = !{i64 1234}

;--- profile_unreachable.yaml
- Guid: 1234
  Counters: [2, 1, 1, 2]