; RUN: llc  -mtriple=x86_64-- < %s

; Ensure that MergeConsecutiveStores doesn't crash when dealing with
; i1 operands.

%struct.X = type { i1, i1 }

@b = common global %struct.X zeroinitializer, align 4

define void @foo() {
entry:
  store i1 0, ptr @b, align 4
  store i1 0, ptr getelementptr inbounds (%struct.X, ptr @b, i64 0, i32 1), align 1
  ret void
}
