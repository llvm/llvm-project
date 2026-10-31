; RUN: opt %loadNPMPolly -passes=polly -S %s | FileCheck %s
;
; https://github.com/llvm/llvm-project/issues/205732
; When DeLICM maps a scalar to an array element, the new access relation
; may only be valid within the DefinedBehaviorContext (i.e., for parameter
; values where the original program has no undefined behavior). Code
; generation must not assert failure for such partial-domain read accesses.
;
; CHECK: polly.start:

define void @foo(i32 %w, ptr %dst, i64 %n) {
entry:
  br label %for.outer

for.outer:
  %i = phi i64 [ %i.next, %for.mid.exit ], [ 0, %entry ]
  br label %for.mid

for.mid:
  br label %for.inner

for.inner:
  %0 = phi i32 [ 0, %for.mid ], [ 1, %for.inner ]
  %1 = load i16, ptr null, align 2
  %cond1 = icmp eq i32 0, %w
  br i1 %cond1, label %for.inner.exit, label %for.inner

for.inner.exit:
  br i1 true, label %for.mid.exit, label %for.mid

for.mid.exit:
  %ptr = getelementptr [4 x i8], ptr %dst, i64 %i
  store i32 %0, ptr %ptr, align 4
  %i.next = add i64 %i, 1
  %cond2 = icmp eq i64 %i, %n
  br i1 %cond2, label %for.outer.exit, label %for.outer

for.outer.exit:
  ret void
}
