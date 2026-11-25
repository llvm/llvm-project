; NOTE: Do not autogenerate.
; Check the default IR, forced candidate remarks, and selected bit-test and normalization instructions.
; RUN: opt -passes=slp-vectorizer -mcpu=x86-64 -verify-each -S < %s -o %t.ll
; RUN: FileCheck %s --check-prefix=IR < %t.ll
; RUN: llc -mcpu=x86-64 < %t.ll | FileCheck %s --check-prefix=ASM
; RUN: opt -passes=slp-vectorizer -mcpu=x86-64 -slp-threshold=-100 -S < %s -o %t.forced.ll -pass-remarks-output=%t.yaml -pass-remarks-filter=slp-vectorizer
; RUN: FileCheck %s --check-prefix=COST < %t.yaml
; RUN: llc -mcpu=x86-64 < %t.forced.ll | FileCheck %s --check-prefix=FORCED
;
; https://github.com/llvm/llvm-project/issues/228871

target triple = "x86_64-unknown-linux-gnu"

declare void @observe_left(i1)
declare void @observe_right(i1)

; IR-LABEL: @store_lanes(
; IR-NOT: icmp eq <2 x i8>
; IR-COUNT-2: icmp eq i8
; IR: ret
; COST: Name: VectorizedList
; COST-NEXT: Function: store_lanes
; COST: - Cost: '3'
; COST: ...
; ASM-LABEL: store_lanes:
; ASM-NOT: movmsk
; ASM: retq
; FORCED-LABEL: store_lanes:
; FORCED: movmskpd
; FORCED-DAG: shrb
; FORCED-DAG: andl $1
; FORCED: movb
define void @store_lanes(ptr noalias %a, ptr noalias %b, ptr %out, ptr %out1) minsize optsize {
entry:
  %a1 = getelementptr inbounds i8, ptr %a, i64 1
  %b1 = getelementptr inbounds i8, ptr %b, i64 1
  %a0v = load i8, ptr %a, align 1
  %a1v = load i8, ptr %a1, align 1
  %b0v = load i8, ptr %b, align 1
  %b1v = load i8, ptr %b1, align 1
  %lane0 = icmp eq i8 %a0v, %b0v
  %lane1 = icmp eq i8 %a1v, %b1v
  store volatile i1 %lane0, ptr %out
  store volatile i1 %lane1, ptr %out1
  ret void
}

; IR-LABEL: @select_lane(
; IR-NOT: icmp eq <2 x i8>
; IR-COUNT-2: icmp eq i8
; IR: ret
; COST: Name: VectorizedList
; COST-NEXT: Function: select_lane
; COST: - Cost: '2'
; COST: ...
; ASM-LABEL: select_lane:
; ASM-NOT: movmsk
; ASM: retq
; FORCED-LABEL: select_lane:
; FORCED: movmskpd
; FORCED-NOT: shrb
; FORCED: testb $2
; FORCED: cmov
define i1 @select_lane(ptr noalias %a, ptr noalias %b, ptr %o, i32 %x, i32 %y) minsize optsize {
entry:
  %a1 = getelementptr inbounds i8, ptr %a, i64 1
  %b1 = getelementptr inbounds i8, ptr %b, i64 1
  %a0v = load i8, ptr %a, align 1
  %a1v = load i8, ptr %a1, align 1
  %b0v = load i8, ptr %b, align 1
  %b1v = load i8, ptr %b1, align 1
  %lane0 = icmp eq i8 %a0v, %b0v
  %lane1 = icmp eq i8 %a1v, %b1v
  %sel = select i1 %lane1, i32 %x, i32 %y
  store i32 %sel, ptr %o
  ret i1 %lane0
}

; IR-LABEL: @branch_lane(
; IR-NOT: icmp eq <2 x i8>
; IR-COUNT-2: icmp eq i8
; IR: ret
; COST: Name: VectorizedList
; COST-NEXT: Function: branch_lane
; COST: - Cost: '2'
; COST: ...
; ASM-LABEL: branch_lane:
; ASM-NOT: movmsk
; ASM: retq
; FORCED-LABEL: branch_lane:
; FORCED: movmskpd
; FORCED-NOT: shrb
; FORCED: testb $2
; FORCED: j{{[en]+}}
define i1 @branch_lane(ptr noalias %a, ptr noalias %b, i1 %x) minsize optsize {
entry:
  %a1 = getelementptr inbounds i8, ptr %a, i64 1
  %b1 = getelementptr inbounds i8, ptr %b, i64 1
  %a0v = load i8, ptr %a, align 1
  %a1v = load i8, ptr %a1, align 1
  %b0v = load i8, ptr %b, align 1
  %b1v = load i8, ptr %b1, align 1
  %lane0 = icmp eq i8 %a0v, %b0v
  %lane1 = icmp eq i8 %a1v, %b1v
  call void @observe_left(i1 %lane0)
  br i1 %lane1, label %l, label %r
l:
  call void @observe_right(i1 %x)
  ret i1 true
r:
  ret i1 false
}
