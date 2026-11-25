; NOTE: Do not autogenerate.
; Keep the selected IR, YAML costs and ABI lowering checks aligned across the default and forced runs.
; RUN: opt -passes=slp-vectorizer -mcpu=x86-64 -verify-each -S < %s -o %t.ll -pass-remarks-output=%t.yaml -pass-remarks-filter=slp-vectorizer
; RUN: FileCheck %s --check-prefix=IR < %t.ll
; RUN: FileCheck %s --check-prefix=COST < %t.yaml
; RUN: opt -passes=slp-vectorizer -slp-threshold=-100 -mcpu=x86-64 -S < %s -o %t.forced.ll -pass-remarks-output=%t.forced.yaml -pass-remarks-filter=slp-vectorizer
; RUN: FileCheck %s --check-prefix=FORCED < %t.forced.ll
; RUN: FileCheck %s --check-prefix=FORCED-COST < %t.forced.yaml
; RUN: llc -mcpu=x86-64 < %t.ll | FileCheck %s --check-prefix=ASM
; RUN: llc -mcpu=x86-64 < %t.forced.ll | FileCheck %s --check-prefix=FORCED-ASM
;
; https://github.com/llvm/llvm-project/issues/228871

target triple = "x86_64-unknown-linux-gnu"

declare void @observe(i1 zeroext, i1 zeroext)
declare void @observe_three(i1 zeroext, i1 zeroext, i1 zeroext)
declare void @observe_one(i1 zeroext)

; IR-LABEL: define void @one_user_two_lanes(
; IR-NOT: icmp eq <2 x i8>
; IR-COUNT-2: icmp eq i8
; IR: call void @observe
;
; COST: Name: NotBeneficial
; COST-NEXT: Function: one_user_two_lanes
; COST: - Cost: '0'
; FORCED-COST: Name: VectorizedList
; FORCED-COST-NEXT: Function: one_user_two_lanes
; FORCED-COST: - Cost: '3'
; FORCED-COST: ...
; ASM-LABEL: one_user_two_lanes:
; ASM-NOT: movmsk
; ASM: callq
; FORCED-ASM-LABEL: one_user_two_lanes:
; FORCED-ASM: movmskpd
; FORCED-ASM-DAG: shrb
; FORCED-ASM-DAG: andl $1
; FORCED-ASM: movzbl
; FORCED-ASM: callq
; COST: ...
define void @one_user_two_lanes(ptr noalias noundef readonly align 8 dereferenceable(72) %a, ptr noalias noundef readonly align 8 dereferenceable(72) %b) {
entry:
  %a0 = getelementptr inbounds i8, ptr %a, i64 56
  %a1 = getelementptr inbounds i8, ptr %a, i64 57
  %b0 = getelementptr inbounds i8, ptr %b, i64 56
  %b1 = getelementptr inbounds i8, ptr %b, i64 57
  %a0v = load i8, ptr %a0, align 8
  %a1v = load i8, ptr %a1, align 1
  %b0v = load i8, ptr %b0, align 8
  %b1v = load i8, ptr %b1, align 1
  %lane0 = icmp eq i8 %a0v, %b0v
  %lane1 = icmp eq i8 %a1v, %b1v
  call void @observe(i1 zeroext %lane0, i1 zeroext %lane1)
  ret void
}

; IR-LABEL: define void @repeated_call_operand(
; IR: [[MASK:%.*]] = icmp eq <2 x i8>
; IR: [[LO:%.*]] = extractelement <2 x i1> [[MASK]], i64 0
; IR: [[HI:%.*]] = extractelement <2 x i1> [[MASK]], i64 1
; IR: call void @observe_three(i1 zeroext [[LO]], i1 zeroext [[HI]], i1 zeroext [[LO]])
;
; COST: Name: VectorizedList
; COST-NEXT: Function: repeated_call_operand
; COST: - Cost: '-1'
; COST: ...
define void @repeated_call_operand(ptr noalias noundef readonly align 8 dereferenceable(72) %a, ptr noalias noundef readonly align 8 dereferenceable(72) %b) {
entry:
  %a0 = getelementptr inbounds i8, ptr %a, i64 56
  %a1 = getelementptr inbounds i8, ptr %a, i64 57
  %b0 = getelementptr inbounds i8, ptr %b, i64 56
  %b1 = getelementptr inbounds i8, ptr %b, i64 57
  %a0v = load i8, ptr %a0, align 8
  %a1v = load i8, ptr %a1, align 1
  %b0v = load i8, ptr %b0, align 8
  %b1v = load i8, ptr %b1, align 1
  %lane0 = icmp eq i8 %a0v, %b0v
  %lane1 = icmp eq i8 %a1v, %b1v
  call void @observe_three(i1 zeroext %lane0, i1 zeroext %lane1, i1 zeroext %lane0)
  ret void
}

; FORCED-COST: Name: StoresVectorized
; FORCED-COST-NEXT: Function: different_producers
; FORCED-COST: - Cost: '5'
; FORCED-COST: ...
; FORCED-LABEL: define void @different_producers(
; FORCED: [[CMP0:%.*]] = icmp eq <2 x i8>
; FORCED: [[CMP1:%.*]] = icmp eq <2 x i8>
; FORCED: [[LO:%.*]] = extractelement <2 x i1> [[CMP1]], i64 0
; FORCED: [[HI:%.*]] = extractelement <2 x i1> [[CMP0]], i64 1
; FORCED: call void @observe(i1 zeroext [[HI]], i1 zeroext [[LO]])
define void @different_producers(ptr noalias %out, ptr noalias %a, ptr noalias %b,
                         ptr noalias %c, ptr noalias %d,
                         i32 %v, i32 %w, i32 %z) {
entry:
  %a1 = getelementptr inbounds i8, ptr %a, i64 1
  %b1 = getelementptr inbounds i8, ptr %b, i64 1
  %c1 = getelementptr inbounds i8, ptr %c, i64 1
  %d1 = getelementptr inbounds i8, ptr %d, i64 1
  %out1 = getelementptr inbounds i32, ptr %out, i64 1
  %a0v = load i8, ptr %a, align 1
  %a1v = load i8, ptr %a1, align 1
  %b0v = load i8, ptr %b, align 1
  %b1v = load i8, ptr %b1, align 1
  %c0v = load i8, ptr %c, align 1
  %c1v = load i8, ptr %c1, align 1
  %d0v = load i8, ptr %d, align 1
  %d1v = load i8, ptr %d1, align 1
  %x0 = icmp eq i8 %a0v, %b0v
  %x1 = icmp eq i8 %a1v, %b1v
  %y0 = icmp eq i8 %c0v, %d0v
  %y1 = icmp eq i8 %c1v, %d1v
  %s0 = select i1 %x0, i32 %v, i32 %w
  %s1 = select i1 %x1, i32 %v, i32 %w
  %r0 = select i1 %y0, i32 %z, i32 %s0
  %r1 = select i1 %y1, i32 %z, i32 %s1
  store i32 %r0, ptr %out, align 4
  store i32 %r1, ptr %out1, align 4
  call void @observe(i1 zeroext %x1, i1 zeroext %y0)
  ret void
}

; FORCED-COST: Name: VectorizedList
; FORCED-COST-NEXT: Function: separate_calls
; FORCED-COST: - Cost: '3'
; FORCED-COST: ...
; IR-LABEL: define void @separate_calls(
; IR-NOT: icmp eq <2 x i8>
; IR-COUNT-2: icmp eq i8
; IR: call void @observe_one
; IR: call void @observe_one
define void @separate_calls(ptr noalias noundef readonly align 8 dereferenceable(72) %a, ptr noalias noundef readonly align 8 dereferenceable(72) %b) {
entry:
  %a0 = getelementptr inbounds i8, ptr %a, i64 56
  %a1 = getelementptr inbounds i8, ptr %a, i64 57
  %b0 = getelementptr inbounds i8, ptr %b, i64 56
  %b1 = getelementptr inbounds i8, ptr %b, i64 57
  %a0v = load i8, ptr %a0, align 8
  %a1v = load i8, ptr %a1, align 1
  %b0v = load i8, ptr %b0, align 8
  %b1v = load i8, ptr %b1, align 1
  %lane0 = icmp eq i8 %a0v, %b0v
  %lane1 = icmp eq i8 %a1v, %b1v
  call void @observe_one(i1 zeroext %lane0)
  call void @observe_one(i1 zeroext %lane1)
  ret void
}
