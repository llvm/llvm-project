; Import hints follow actual inlines. They become critical edges only when the
; original body has been deleted, even with multiple callers or nested inlines.
;
; RUN: opt -passes=always-inline -S %s -o %t.ll
; RUN: FileCheck %s --check-prefix=INLINE < %t.ll
; RUN: opt -module-summary %t.ll -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefix=SUMMARY --implicit-check-not=sample.pending.imports
; The production ThinLTO writer also consumes the temporary metadata.
; RUN: opt -passes=always-inline -thinlto-bc %s -o %t.thin.bc
; RUN: llvm-dis %t.thin.bc -o - | FileCheck %s --check-prefix=SUMMARY --implicit-check-not=sample.pending.imports
;
; R is inlined into both A and B; calling it twice in A does not duplicate entries.
; H stays alive, but its previously inlined leaf does not. Keep the leaf's key
; when H is subsequently inlined into A and B.
; INLINE-LABEL: define void @A()
; INLINE-SAME: !sample.pending.imports ![[P:[0-9]+]]
; INLINE-LABEL: define void @B()
; INLINE-SAME: !sample.pending.imports ![[P]]
; INLINE-NOT: define internal void @R
; INLINE-NOT: define internal void @leaf
; INLINE: ![[P]] = !{![[R:[0-9]+]], ![[NESTED:[0-9]+]]}
; INLINE: ![[R]] = !{i64 1, !{{[0-9]+}}}
; INLINE: ![[NESTED]] = !{![[H:[0-9]+]], ![[LEAF:[0-9]+]]}
; INLINE: ![[H]] = !{i64 2, !{{[0-9]+}}}
; INLINE: ![[LEAF]] = !{i64 3, !{{[0-9]+}}}
;
; SUMMARY-DAG: ^[[EXT:[0-9]+]] = gv: (guid: 101)
; SUMMARY-DAG: ^[[LEAFEXT:[0-9]+]] = gv: (guid: 103)
; SUMMARY-DAG: name: "A", {{.*}}calls: ((callee: ^[[LEAFEXT]], hotness: critical), (callee: ^[[EXT]], hotness: critical))
; SUMMARY-DAG: name: "B", {{.*}}calls: ((callee: ^[[LEAFEXT]], hotness: critical), (callee: ^[[EXT]], hotness: critical))

source_filename = "imports.c"

define void @A() {
  call void @R()
  call void @R()
  call void @H()
  ret void
}

define void @B() {
  call void @R()
  call void @H()
  ret void
}

define internal void @R() alwaysinline !guid !0 !prof !3 !sample.pending.imports !6 {
  ret void
}

; Put leaf before H so the always-inliner processes this inner inline first.
define internal void @leaf() alwaysinline !guid !2 !prof !5 !sample.pending.imports !8 {
  ret void
}

define void @H() alwaysinline !guid !1 !prof !4 !sample.pending.imports !7 {
  call void @leaf()
  ret void
}

!0 = !{i64 1}
!1 = !{i64 2}
!2 = !{i64 3}
!3 = !{!"function_entry_count", i64 100, i64 101}
!4 = !{!"function_entry_count", i64 100, i64 102}
!5 = !{!"function_entry_count", i64 100, i64 103}
!6 = !{i64 1, !3}
!7 = !{i64 2, !4}
!8 = !{i64 3, !5}
