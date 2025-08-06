; Test that verifies functionality for -opt-bisect-skip

; RUN: opt -O1 -opt-bisect-skip=3 -opt-bisect-skip=7 %s 2>&1 | FileCheck %s --check-prefix=CHECK-DISABLE-PASS
; CHECK-DISABLE-PASS: BISECT: running pass (1) annotation2metadata on [module]
; CHECK-DISABLE-PASS: BISECT: running pass (2) forceattrs on [module]
; CHECK-DISABLE-PASS: BISECT: NOT running pass (3) inferattrs on [module]
; CHECK-DISABLE-PASS: BISECT: running pass (4) lower-expect on foo
; CHECK-DISABLE-PASS: BISECT: running pass (5) simplifycfg on foo
; CHECK-DISABLE-PASS: BISECT: running pass (6) sroa on foo
; CHECK-DISABLE-PASS: BISECT: NOT running pass (7) early-cse on foo
; CHECK-DISABLE-PASS: BISECT: running pass (8) openmp-opt on [module]

define void @foo() {
  ret void
}