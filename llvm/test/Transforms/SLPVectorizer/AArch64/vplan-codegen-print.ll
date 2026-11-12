; REQUIRES: asserts
; RUN: opt -passes=slp-vectorizer -mtriple=aarch64 -slp-threshold=-10 \
; RUN:   -slp-runtime-alias-checks-max-scalar-cost-percent=1000 \
; RUN:   -slp-use-vplan-codegen -debug-only=SLP -disable-output %s 2>&1 | FileCheck %s

define void @print_vplan(ptr %a, ptr %b, ptr %c) {
; CHECK-LABEL: SLP: VPlan for tree:
; CHECK-NEXT:  VPlan 'SLP tree for UF>=1' {
; CHECK-EMPTY:
; CHECK-NEXT:  {{.+}}:
; CHECK-NEXT:    IR   %x0 = load i32, ptr %a, align 4
; CHECK-NEXT:    WIDEN ir<[[X:%.+]]> = load ir<%a>
; CHECK-NEXT:    WIDEN ir<[[Y:%.+]]> = load ir<%b>
; CHECK-NEXT:    WIDEN ir<[[S:%.+]]> = add nsw ir<[[X]]>, ir<[[Y]]>
; CHECK-NEXT:    WIDEN store ir<%c>, ir<[[S]]>
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
;
  %a1 = getelementptr i32, ptr %a, i64 1
  %b1 = getelementptr i32, ptr %b, i64 1
  %c1 = getelementptr i32, ptr %c, i64 1
  %x0 = load i32, ptr %a
  %x1 = load i32, ptr %a1
  %y0 = load i32, ptr %b
  %y1 = load i32, ptr %b1
  %s0 = add nsw i32 %x0, %y0
  %s1 = add nsw i32 %x1, %y1
  store i32 %s0, ptr %c
  store i32 %s1, ptr %c1
  ret void
}

declare void @may_write()

; The loads must be placed before the call, which may write memory, and the
; store after it.
define void @call_between_loads_and_store(ptr %a, ptr %b, ptr %c) {
; CHECK-LABEL: SLP: Analyzing blocks in call_between_loads_and_store.
; CHECK:       SLP: VPlan for tree:
; CHECK-NEXT:  VPlan 'SLP tree for UF>=1' {
; CHECK-EMPTY:
; CHECK-NEXT:  {{.+}}:
; CHECK-NEXT:    IR   %x0 = load i32, ptr %a, align 4
; CHECK-NEXT:    WIDEN ir<%x0> = load ir<%a>
; CHECK-NEXT:    WIDEN ir<%y0> = load ir<%b>
; CHECK-NEXT:    WIDEN ir<%s0> = add nsw ir<%x0>, ir<%y0>
; CHECK-NEXT:    IR   store i32 %s0, ptr %c, align 4
; CHECK-NEXT:    WIDEN store ir<%c>, ir<%s0>
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
;
  %a1 = getelementptr i32, ptr %a, i64 1
  %b1 = getelementptr i32, ptr %b, i64 1
  %c1 = getelementptr i32, ptr %c, i64 1
  %x0 = load i32, ptr %a
  %x1 = load i32, ptr %a1
  %y0 = load i32, ptr %b
  %y1 = load i32, ptr %b1
  %s0 = add nsw i32 %x0, %y0
  %s1 = add nsw i32 %x1, %y1
  call void @may_write()
  store i32 %s0, ptr %c
  store i32 %s1, ptr %c1
  ret void
}

; Extracts for external uses are not supported yet.
define void @external_use(ptr %a, ptr %b, ptr %c, ptr %out) {
; CHECK-LABEL: SLP: Analyzing blocks in external_use.
; CHECK-NOT:   SLP: VPlan for tree
;
  %a1 = getelementptr i32, ptr %a, i64 1
  %b1 = getelementptr i32, ptr %b, i64 1
  %c1 = getelementptr i32, ptr %c, i64 1
  %x0 = load i32, ptr %a
  %x1 = load i32, ptr %a1
  %y0 = load i32, ptr %b
  %y1 = load i32, ptr %b1
  %s0 = add i32 %x0, %y0
  %s1 = add i32 %x1, %y1
  store i32 %s0, ptr %c
  store i32 %s1, ptr %c1
  store i32 %s1, ptr %out
  ret void
}

; Gathered operands are not supported yet.
define void @gather_operand(ptr %a, ptr %c, i32 %y0, i32 %y1) {
; CHECK-LABEL: SLP: Analyzing blocks in gather_operand.
; CHECK-NOT:   SLP: VPlan for tree
;
  %a1 = getelementptr i32, ptr %a, i64 1
  %c1 = getelementptr i32, ptr %c, i64 1
  %x0 = load i32, ptr %a
  %x1 = load i32, ptr %a1
  %s0 = add i32 %x0, %y0
  %s1 = add i32 %x1, %y1
  store i32 %s0, ptr %c
  store i32 %s1, ptr %c1
  ret void
}

; Casts are not supported yet.
define void @unsupported_opcode(ptr %a, ptr %c) {
; CHECK-LABEL: SLP: Analyzing blocks in unsupported_opcode.
; CHECK-NOT:   SLP: VPlan for tree
;
  %a1 = getelementptr i32, ptr %a, i64 1
  %c1 = getelementptr i64, ptr %c, i64 1
  %x0 = load i32, ptr %a
  %x1 = load i32, ptr %a1
  %s0 = zext i32 %x0 to i64
  %s1 = zext i32 %x1 to i64
  store i64 %s0, ptr %c
  store i64 %s1, ptr %c1
  ret void
}

; Alternate opcodes are not supported yet.
define void @alternate_opcodes(ptr %a, ptr %b, ptr %c) {
; CHECK-LABEL: SLP: Analyzing blocks in alternate_opcodes.
; CHECK-NOT:   SLP: VPlan for tree
;
  %a1 = getelementptr i32, ptr %a, i64 1
  %b1 = getelementptr i32, ptr %b, i64 1
  %c1 = getelementptr i32, ptr %c, i64 1
  %x0 = load i32, ptr %a
  %x1 = load i32, ptr %a1
  %y0 = load i32, ptr %b
  %y1 = load i32, ptr %b1
  %s0 = add i32 %x0, %y0
  %s1 = sub i32 %x1, %y1
  store i32 %s0, ptr %c
  store i32 %s1, ptr %c1
  ret void
}

; Reordered lanes are not supported yet.
define void @reordered_lanes(ptr %a, ptr %b, ptr %c) {
; CHECK-LABEL: SLP: Analyzing blocks in reordered_lanes.
; CHECK-NOT:   SLP: VPlan for tree
;
  %a1 = getelementptr i32, ptr %a, i64 1
  %b1 = getelementptr i32, ptr %b, i64 1
  %c1 = getelementptr i32, ptr %c, i64 1
  %x0 = load i32, ptr %a
  %x1 = load i32, ptr %a1
  %y0 = load i32, ptr %b
  %y1 = load i32, ptr %b1
  %s0 = add i32 %x1, %y1
  %s1 = add i32 %x0, %y0
  store i32 %s0, ptr %c
  store i32 %s1, ptr %c1
  ret void
}

; Entries outside the root's block are not supported yet.
define void @operands_in_other_block(ptr %a, ptr %b, ptr %c) {
; CHECK-LABEL: SLP: Analyzing blocks in operands_in_other_block.
; CHECK-NOT:   SLP: VPlan for tree
;
entry:
  %a1 = getelementptr i32, ptr %a, i64 1
  %b1 = getelementptr i32, ptr %b, i64 1
  %x0 = load i32, ptr %a
  %x1 = load i32, ptr %a1
  %y0 = load i32, ptr %b
  %y1 = load i32, ptr %b1
  br label %next

next:
  %c1 = getelementptr i32, ptr %c, i64 1
  %s0 = add i32 %x0, %y0
  %s1 = add i32 %x1, %y1
  store i32 %s0, ptr %c
  store i32 %s1, ptr %c1
  ret void
}

; Reductions are not supported yet.
define i32 @reduction(ptr %a) {
; CHECK-LABEL: SLP: Analyzing blocks in reduction.
; CHECK-NOT:   SLP: VPlan for tree
;
  %a1 = getelementptr i32, ptr %a, i64 1
  %a2 = getelementptr i32, ptr %a, i64 2
  %a3 = getelementptr i32, ptr %a, i64 3
  %x0 = load i32, ptr %a
  %x1 = load i32, ptr %a1
  %x2 = load i32, ptr %a2
  %x3 = load i32, ptr %a3
  %r0 = add i32 %x0, %x1
  %r1 = add i32 %r0, %x2
  %r2 = add i32 %r1, %x3
  ret i32 %r2
}

; Versioning for runtime alias checks is not supported yet.
define void @runtime_alias_checks(ptr %a, ptr %b, ptr %c) {
; CHECK-LABEL: SLP: Analyzing blocks in runtime_alias_checks.
; CHECK-NOT:   SLP: VPlan for tree
;
  %a1 = getelementptr i32, ptr %a, i64 1
  %a2 = getelementptr i32, ptr %a, i64 2
  %a3 = getelementptr i32, ptr %a, i64 3
  %b1 = getelementptr i32, ptr %b, i64 1
  %b2 = getelementptr i32, ptr %b, i64 2
  %b3 = getelementptr i32, ptr %b, i64 3
  %c1 = getelementptr i32, ptr %c, i64 1
  %c2 = getelementptr i32, ptr %c, i64 2
  %c3 = getelementptr i32, ptr %c, i64 3
  %x0 = load i32, ptr %a
  %y0 = load i32, ptr %b
  %s0 = mul i32 %x0, %y0
  store i32 %s0, ptr %c
  %x1 = load i32, ptr %a1
  %y1 = load i32, ptr %b1
  %s1 = mul i32 %x1, %y1
  store i32 %s1, ptr %c1
  %x2 = load i32, ptr %a2
  %y2 = load i32, ptr %b2
  %s2 = mul i32 %x2, %y2
  store i32 %s2, ptr %c2
  %x3 = load i32, ptr %a3
  %y3 = load i32, ptr %b3
  %s3 = mul i32 %x3, %y3
  store i32 %s3, ptr %c3
  ret void
}
