; NOTE: Do not autogenerate. Check loop metadata across a batch of edge splits.
; RUN: split-file %s %t
; RUN: opt -S -verify-each -passes=pgo-instr-gen %t/input.ll | FileCheck %s
; RUN: llvm-profdata merge %t/profile.txt -o %t.profdata
; RUN: opt -S -verify-each -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata %t/input.ll | FileCheck %s
; RUN: opt -S -verify-each -passes='require<domtree>,break-crit-edges,verify<domtree>' %t/input.ll | FileCheck %s
;
; Splits both before the first annotated loop and between annotated loops must
; be reflected in the temporary dominator tree used for metadata ownership.
;
; CHECK-LABEL: define void @loops(
; CHECK: first:
; CHECK: br i1 %again, label %[[FIRST:[^, ]+]], label %{{[^, ]+}}{{(, !prof ![0-9]+)?}}{{$}}
; CHECK: [[FIRST]]:
; CHECK: br label %first{{$}}
; CHECK: second:
; CHECK: br i1 %again, label %[[SECOND:[^, ]+]], label %{{[^, ]+}}{{(, !prof ![0-9]+)?}}{{$}}
; CHECK: [[SECOND]]:
; CHECK: br label %second, !llvm.loop ![[ID:[0-9]+]]
; CHECK: third:
; CHECK: br i1 %again, label %[[THIRD:[^, ]+]], label %{{[^, ]+}}{{(, !prof ![0-9]+)?}}{{$}}
; CHECK: [[THIRD]]:
; CHECK: br label %third{{$}}
; CHECK: fourth:
; CHECK: br i1 %again, label %[[FOURTH:[^, ]+]], label %exit{{(, !prof ![0-9]+)?}}{{$}}
; CHECK: [[FOURTH]]:
; CHECK: br label %fourth, !llvm.loop ![[ID]]
;
;--- input.ll
define void @loops(i1 %again) {
entry:
  br label %first
first:
  br i1 %again, label %first, label %second
second:
  br i1 %again, label %second, label %third, !llvm.loop !0
third:
  br i1 %again, label %third, label %fourth
fourth:
  br i1 %again, label %fourth, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}

;--- profile.txt
:ir
loops
844982797682130989
5
20
20
20
20
10
