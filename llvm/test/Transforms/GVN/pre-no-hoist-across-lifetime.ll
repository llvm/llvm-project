; RUN: opt < %s -passes=gvn -S | FileCheck %s

; Regression test for
; https://github.com/llvm/llvm-project/issues/194940
; GVN Load PRE must not hoist a load across @llvm.lifetime.start of the
; alloca it accesses (partial overlap).

target datalayout = "e-p:64:64-i64:64-n32:64-S128"

%Struct = type { [80 x i8], i16 }

declare void @opaque(ptr)
declare void @llvm.lifetime.start.p0(ptr captures(none))
declare void @llvm.lifetime.end.p0(ptr captures(none))

; The load of %s through a GEP must NOT be hoisted into %cold, which runs
; before the lifetime.start in %merge; the load has to stay after it.
define void @dont_hoist_across_lifetime_start(ptr %out, i1 %cond) {
; CHECK-LABEL: @dont_hoist_across_lifetime_start(
; CHECK:       cold:
; CHECK-NOT:     load
; CHECK:         br label %merge
; CHECK:       merge:
; CHECK-NEXT:    call void @llvm.lifetime.start
; CHECK:         load i16
entry:
  %s = alloca %Struct, align 8
  br i1 %cond, label %merge, label %cold

cold:
  ; Keep %cold from being eliminated — the side effect matters so that PRE
  ; has a non-trivial predecessor to hoist into.
  store ptr null, ptr inttoptr (i64 8 to ptr), align 8
  br label %merge

merge:
  call void @llvm.lifetime.start.p0(ptr %s)
  %s.gep = getelementptr inbounds i8, ptr %s, i64 80
  %v = load i16, ptr %s.gep, align 8
  store i16 %v, ptr %out, align 2
  call void @opaque(ptr %s)
  call void @llvm.lifetime.end.p0(ptr %s)
  ret void
}

; Sanity check: for a load that is dominated by lifetime.start on every
; incoming edge, GVN can still eliminate the redundant loads across the join
; (nothing about the fix should block that).
define i16 @eliminate_redundant_load_after_lifetime_start(i1 %cond) {
; CHECK-LABEL: @eliminate_redundant_load_after_lifetime_start(
; CHECK:       join:
; CHECK-NEXT:    call void @llvm.lifetime.end
; CHECK-NEXT:    ret i16 42
entry:
  %s = alloca %Struct, align 8
  call void @llvm.lifetime.start.p0(ptr %s)
  %s.gep = getelementptr inbounds i8, ptr %s, i64 80
  store i16 42, ptr %s.gep, align 8
  br i1 %cond, label %a, label %b

a:
  %va = load i16, ptr %s.gep, align 8
  br label %join

b:
  %vb = load i16, ptr %s.gep, align 8
  br label %join

join:
  %v = phi i16 [ %va, %a ], [ %vb, %b ]
  call void @llvm.lifetime.end.p0(ptr %s)
  ret i16 %v
}
