; RUN: opt -S -passes='ipsccp,function(loop(loop-rotate))' -force-specialization < %s | FileCheck %s --check-prefix=NO-LATE-IPSCCP
; RUN: opt -S -passes='ipsccp,function(loop(loop-rotate)),ipsccp' -force-specialization < %s | FileCheck %s --check-prefix=WITH-LATE-IPSCCP

@external_cond = external global i1, align 1

define void @top_level_caller() {
; WITH-LATE-IPSCCP-LABEL: define void @top_level_caller(
; WITH-LATE-IPSCCP:        call void @digits_2.specialized.{{[0-9]+}}(

; NO-LATE-IPSCCP-LABEL:   define void @top_level_caller(
; NO-LATE-IPSCCP-NOT:       @digits_2.specialized
entry:
  %temp = alloca i32, align 4
  store i32 2, ptr %temp, align 4
  %cond = load i1, ptr @external_cond, align 1
  %idx = select i1 %cond, i32 0, i32 99
  call void @digits_2(ptr %temp, i32 %idx)
  ret void
}

define internal void @digits_2(ptr %arg1, i32 %loop_idx) {
; NO-LATE-IPSCCP-LABEL: define internal void @digits_2(
; NO-LATE-IPSCCP:        %val1 = load i32, ptr %arg1
; NO-LATE-IPSCCP:        br label %loop.body

; WITH-LATE-IPSCCP-LABEL: define internal void @digits_2.specialized.{{[0-9]+}}(
entry:
  %temp_alloc = alloca i32, align 4
  br label %loop.header

loop.header:
  %ptr_input = phi ptr [ %arg1, %entry ], [ %arrayidx, %loop.body ]
  %iv = phi i32 [ %loop_idx, %entry ], [ %iv.next, %loop.body ]
  %val = load i32, ptr %ptr_input, align 4
  %cmp = icmp slt i32 %iv, 1
  br i1 %cmp, label %loop.body, label %exit

loop.body:
  %idxprom = sext i32 %iv to i64
  %arrayidx = getelementptr inbounds i32, ptr %ptr_input, i64 %idxprom
  store i32 %val, ptr %arrayidx, align 4
  %iv.next = add nsw i32 %iv, 1
  call void @digits_2(ptr %temp_alloc, i32 %iv.next)
  br label %loop.header

exit:
  ret void
}
