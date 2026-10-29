; NOTE: Do not autogenerate. Check wave event identities and measured/unknown\n; state across composed CFG rewrites without unrelated instruction checks.\n; RUN: split-file %s %t
; RUN: opt -S -passes='loop(loop-unroll-full),verify' %t/simple.ll | FileCheck %s --check-prefix=UNROLL
; RUN: opt -S -passes='loop(loop-unroll-full),simplifycfg,verify' %t/simple.ll | FileCheck %s --check-prefix=MERGE
; RUN: opt -S -passes='loop(loop-unroll-full),simplifycfg,verify' %t/nested.ll | FileCheck %s --check-prefix=NESTED
; RUN: opt -S -passes='loop-unroll,verify' -unroll-count=2 %t/partial.ll | FileCheck %s --check-prefix=PARTIAL
; RUN: opt -S -passes='loop-unroll,verify' -unroll-count=4 %t/early-exit.ll | FileCheck %s --check-prefix=EARLY

;--- simple.ll
;
; Full unrolling replaces a loop event and transplants its cleanup terminator.
; Keep the entry event and its normalization, without retaining the loop count.
; Subsequent merging retains the predecessor's measured event.
;
; UNROLL-LABEL: define amdgpu_kernel void @_Z14divergent_loopPVi(
; UNROLL-SAME: !wave.profile [[TABLE:![0-9]+]]
; UNROLL: entry:
; UNROLL: br label %body, !wave.profile.block [[ENTRY:![0-9]+]]
; UNROLL: body:
; UNROLL-COUNT-4: store volatile i32
; UNROLL: ret void, !wave.profile.block [[BODY:![0-9]+]]
; UNROLL: [[TABLE]] = distinct !{i64 2, i64 2480672276464841217, i64 6, i64 24, i64 0{{(, i64 0)+}}}
; UNROLL: [[ENTRY]] = !{i64 2, i64 2480672276464841217, i64 0, i64 1, i64 3}
; UNROLL: [[BODY]] = !{i64 2, i64 2480672276464841217, i64 3, i64 0}
;
; MERGE-LABEL: define amdgpu_kernel void @_Z14divergent_loopPVi(
; MERGE: entry:
; MERGE-COUNT-4: store volatile i32
; MERGE: ret void, !wave.profile.block [[ENTRY:![0-9]+]]
; MERGE: [[ENTRY]] = !{i64 2, i64 2480672276464841217, i64 0, i64 1}
define amdgpu_kernel void @_Z14divergent_loopPVi(ptr addrspace(1) %out) !wave.profile !0 {
entry:
  br label %body, !wave.profile.block !1
body:
  %i = phi i32 [0, %entry], [%next, %body]
  store volatile i32 %i, ptr addrspace(1) %out
  %next = add nuw nsw i32 %i, 1
  %more = icmp ult i32 %next, 4
  br i1 %more, label %body, label %exit, !wave.profile.block !2
exit:
  ret void, !wave.profile.block !3
}
!0 = !{i64 2, i64 2480672276464841217, i64 6, i64 24, i64 0}
!1 = !{i64 2, i64 2480672276464841217, i64 0, i64 1, i64 1}
!2 = !{i64 2, i64 2480672276464841217, i64 1, i64 1, i64 1, i64 2}
!3 = !{i64 2, i64 2480672276464841217, i64 2, i64 0}

;--- nested.ll
; Preserve the surrounding outer-loop event through inner full unroll and merge.
; NESTED-LABEL: define amdgpu_kernel void @_Z14divergent_loopPVi(
; NESTED-SAME: !wave.profile [[TABLE:![0-9]+]]
; NESTED: entry:
; NESTED: br label %outer, !wave.profile.block [[ENTRY:![0-9]+]]
; NESTED: outer:
; NESTED-COUNT-4: store volatile i32
; NESTED: br i1 %outer.more, label %outer, label %exit, !llvm.loop {{![0-9]+}}, !wave.profile.block [[OUTER:![0-9]+]]
; NESTED: [[TABLE]] = distinct !{i64 2, i64 2480672276464841217, i64 6, i64 66, i64 264, i64 66, i64 0{{(, i64 0)+}}}
; NESTED: [[ENTRY]] = !{i64 2, i64 2480672276464841217, i64 0, i64 1, i64 1}
; NESTED: [[OUTER]] = !{i64 2, i64 2480672276464841217, i64 1, i64 1, i64 1, i64 4}
define amdgpu_kernel void @_Z14divergent_loopPVi(ptr addrspace(1) %out, i32 %n) !wave.profile !0 {
entry:
  br label %outer, !wave.profile.block !1
outer:
  %j = phi i32 [0, %entry], [%j.next, %outer.latch]
  br label %body, !wave.profile.block !2
body:
  %i = phi i32 [0, %outer], [%next, %body]
  store volatile i32 %i, ptr addrspace(1) %out
  %next = add nuw nsw i32 %i, 1
  %more = icmp ult i32 %next, 4
  br i1 %more, label %body, label %outer.latch, !wave.profile.block !3
outer.latch:
  %j.next = add nuw nsw i32 %j, 1
  %outer.more = icmp ult i32 %j.next, %n
  br i1 %outer.more, label %outer, label %exit, !llvm.loop !6, !wave.profile.block !4
exit:
  ret void, !wave.profile.block !5
}
!0 = !{i64 2, i64 2480672276464841217, i64 6, i64 66, i64 264, i64 66, i64 0}
!1 = !{i64 2, i64 2480672276464841217, i64 0, i64 1, i64 1}
!2 = !{i64 2, i64 2480672276464841217, i64 1, i64 1, i64 2}
!3 = !{i64 2, i64 2480672276464841217, i64 2, i64 1, i64 2, i64 3}
!4 = !{i64 2, i64 2480672276464841217, i64 3, i64 1, i64 1, i64 4}
!5 = !{i64 2, i64 2480672276464841217, i64 4, i64 0}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.unroll.disable"}

;--- partial.ll
; The original loop count does not describe the partially unrolled loop.
; Do not revive it when the duplicated identities collapse during merging.
; PARTIAL-LABEL: define amdgpu_kernel void @_Z14divergent_loopPVi(
; PARTIAL: entry:
; PARTIAL: br label %body, !wave.profile.block [[ENTRY:![0-9]+]]
; PARTIAL: body:
; PARTIAL-COUNT-2: store volatile i32
; PARTIAL: br i1 %more.1, label %body, label %exit, !llvm.loop {{![0-9]+}}, !wave.profile.block [[BODY:![0-9]+]]
; PARTIAL: [[ENTRY]] = !{i64 2, i64 2480672276464841217, i64 3, i64 0, i64 4}
; PARTIAL: [[BODY]] = !{i64 2, i64 2480672276464841217, i64 4, i64 0, i64 4, i64 2}
define amdgpu_kernel void @_Z14divergent_loopPVi(ptr addrspace(1) %out) !wave.profile !0 {
entry:
  br label %body, !wave.profile.block !1
body:
  %i = phi i32 [0, %entry], [%next, %body]
  store volatile i32 %i, ptr addrspace(1) %out
  %next = add nuw nsw i32 %i, 1
  %more = icmp ult i32 %next, 8
  br i1 %more, label %body, label %exit, !wave.profile.block !2
exit:
  ret void, !wave.profile.block !3
}
!0 = !{i64 2, i64 2480672276464841217, i64 6, i64 48, i64 0}
!1 = !{i64 2, i64 2480672276464841217, i64 0, i64 1, i64 1}
!2 = !{i64 2, i64 2480672276464841217, i64 1, i64 1, i64 1, i64 2}
!3 = !{i64 2, i64 2480672276464841217, i64 2, i64 0}

;--- early-exit.ll
; A trip-count upper bound is not an exact trip count. Leave copied identities
; ambiguous so the validator rejects them, rather than certifying new events.
; EARLY-LABEL: define amdgpu_kernel void @_Z14divergent_loopPVi(
; EARLY-SAME: !wave.profile [[TABLE:![0-9]+]]
; EARLY: br i1 %keepGoing, label %body.1, label %exit, !wave.profile.block [[COPY:![0-9]+]]
; EARLY: br i1 %keepGoing, label %body.2, label %exit, !wave.profile.block [[COPY]]
; EARLY: br i1 %keepGoing, label %body.3, label %exit, !wave.profile.block [[COPY]]
; EARLY: [[TABLE]] = !{i64 2, i64 2480672276464841217, i64 6, i64 24, i64 0}
; EARLY: [[COPY]] = !{i64 2, i64 2480672276464841217, i64 1, i64 1, i64 1, i64 2}
define amdgpu_kernel void @_Z14divergent_loopPVi(ptr addrspace(1) %out, i1 %keepGoing) !wave.profile !0 {
entry:
  br label %body, !wave.profile.block !1
body:
  %i = phi i32 [0, %entry], [%next, %body]
  store volatile i32 %i, ptr addrspace(1) %out
  %next = add nuw nsw i32 %i, 1
  %more = icmp ult i32 %next, 4
  %again = and i1 %more, %keepGoing
  br i1 %again, label %body, label %exit, !wave.profile.block !2
exit:
  ret void, !wave.profile.block !3
}
!0 = !{i64 2, i64 2480672276464841217, i64 6, i64 24, i64 0}
!1 = !{i64 2, i64 2480672276464841217, i64 0, i64 1, i64 1}
!2 = !{i64 2, i64 2480672276464841217, i64 1, i64 1, i64 1, i64 2}
!3 = !{i64 2, i64 2480672276464841217, i64 2, i64 0}
