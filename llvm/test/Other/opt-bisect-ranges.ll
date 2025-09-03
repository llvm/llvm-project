; Test that verifies functionality for -opt-bisect with range specifications

; Test basic range functionality: run passes 1-3 and 7-8
; RUN: opt -O1 -opt-bisect=1-3,7-8 %s 2>&1 | FileCheck %s --check-prefix=CHECK-RANGES
; CHECK-RANGES: BISECT: running pass (1) annotation2metadata on [module]
; CHECK-RANGES: BISECT: running pass (2) forceattrs on [module]
; CHECK-RANGES: BISECT: running pass (3) inferattrs on [module]
; CHECK-RANGES: BISECT: NOT running pass (4) lower-expect on foo
; CHECK-RANGES: BISECT: NOT running pass (5) simplifycfg on foo
; CHECK-RANGES: BISECT: NOT running pass (6) sroa on foo
; CHECK-RANGES: BISECT: running pass (7) early-cse on foo
; CHECK-RANGES: BISECT: running pass (8) openmp-opt on [module]

; Test single pass selection: run only pass 5
; RUN: opt -O1 -opt-bisect=5 %s 2>&1 | FileCheck %s --check-prefix=CHECK-SINGLE
; CHECK-SINGLE: BISECT: NOT running pass (1) annotation2metadata on [module]
; CHECK-SINGLE: BISECT: NOT running pass (2) forceattrs on [module]
; CHECK-SINGLE: BISECT: NOT running pass (3) inferattrs on [module]
; CHECK-SINGLE: BISECT: NOT running pass (4) lower-expect on foo
; CHECK-SINGLE: BISECT: running pass (5) simplifycfg on foo
; CHECK-SINGLE: BISECT: NOT running pass (6) sroa on foo

; Test backward compatibility: -opt-bisect-limit=3 should be equivalent to -opt-bisect=1-3
; RUN: opt -O1 -opt-bisect-limit=3 %s 2>&1 | FileCheck %s --check-prefix=CHECK-LIMIT
; CHECK-LIMIT: BISECT: running pass (1) annotation2metadata on [module]
; CHECK-LIMIT: BISECT: running pass (2) forceattrs on [module]
; CHECK-LIMIT: BISECT: running pass (3) inferattrs on [module]
; CHECK-LIMIT: BISECT: NOT running pass (4) lower-expect on foo
; CHECK-LIMIT: BISECT: NOT running pass (5) simplifycfg on foo

define void @foo() {
  ret void
}


