;; This test verifies 'auto' hotness threshold when profile summary is available.
;;
;; new PM
; RUN: rm -f %t.yaml %t.hot.yaml
; RUN: opt < %s --disable-output \
; RUN: --passes='inline' \
; RUN: --pass-remarks-output=%t.yaml --pass-remarks-filter='inline' \
; RUN: --pass-remarks-with-hotness
; RUN: FileCheck %s -check-prefix=YAML-PASS < %t.yaml
; RUN: FileCheck %s -check-prefix=YAML-MISS < %t.yaml

;; test 'auto' threshold
; RUN: opt < %s --disable-output --inline-enable-cost-benefit-analysis=0 \
; RUN: --passes='module(print-profile-summary,cgscc(inline))' \
; RUN: --pass-remarks-output=%t.hot.yaml --pass-remarks-filter='inline' \
; RUN: --pass-remarks-with-hotness --pass-remarks-hotness-threshold=auto 2>&1 | FileCheck %s
; RUN: FileCheck %s -check-prefix=YAML-PASS < %t.hot.yaml
; RUN: not FileCheck %s -check-prefix=YAML-MISS < %t.hot.yaml

; RUN: opt < %s --disable-output --inline-enable-cost-benefit-analysis=0  \
; RUN: --passes='module(print-profile-summary,cgscc(inline))' \
; RUN: --pass-remarks=inline --pass-remarks-missed=inline --pass-remarks-analysis=inline \
; RUN: --pass-remarks-with-hotness --pass-remarks-hotness-threshold=auto 2>&1 | FileCheck %s -check-prefix=CHECK-RPASS

; YAML-PASS:      --- !Passed
; YAML-PASS-NEXT: Pass:            inline
; YAML-PASS-NEXT: Name:            Inlined
; YAML-PASS-NEXT: Function:        caller1
; YAML-PASS-NEXT: Hotness:         400

; YAML-MISS:      --- !Missed
; YAML-MISS-NEXT: Pass:            inline
; YAML-MISS-NEXT: Name:            NeverInline
; YAML-MISS-NEXT: Function:        caller2
; YAML-MISS-NEXT: Hotness:         1

; CHECK-RPASS: 'callee1' inlined into 'caller1' with (cost=-30, threshold=4500) (hotness: 400)
; CHECK-RPASS-NOT: 'callee2' not inlined into 'caller2' because it should never be inlined (cost=never): noinline function attribute (hotness: 1)

define void @callee1() !prof !20 {
; CHECK: callee1 :hot
entry:
  ret void
}

; Function Attrs: noinline
define void @callee2() noinline !prof !21 {
; CHECK: callee2 :cold
entry:
  ret void
}

define void @caller1() !prof !20 {
; CHECK: caller1 :hot
entry:
  call void @callee1()
  ret void
}

define void @caller2() !prof !21 {
; CHECK: caller2 :cold
entry:
  call void @callee2()
  ret void
}

!llvm.module.flags = !{!1}
!20 = !{!"function_entry_count", i64 400}
!21 = !{!"function_entry_count", i64 1}

!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 10000}
!5 = !{!"MaxCount", i64 10}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 1000}
!8 = !{!"NumCounts", i64 3}
!9 = !{!"NumFunctions", i64 3}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 100, i32 1}
!13 = !{i32 999000, i64 100, i32 1}
!14 = !{i32 999999, i64 1, i32 2}
