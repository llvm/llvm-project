; RUN: opt < %s -loop-reduce -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; This confirms that NVPTXTTI considers a 64-to-32 integer trunc free. If such
; truncs were not considered free, LSR would promote (int)i as a separate
; induction variable in the following example.
;
;   for (long i = begin; i != end; i += stride)
;     use((int)i);
;
; That would be worthless, because "i" is simulated by two 32-bit registers and
; truncating it to 32-bit is as simple as directly using the register that
; contains the low bits.
define void @trunc_is_free(i64 %begin, i64 %stride, i64 %end) {
; CHECK-LABEL: @trunc_is_free(
entry:
  %cmp.4 = icmp eq i64 %begin, %end
  br i1 %cmp.4, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
; CHECK: for.body:
  %i.05 = phi i64 [ %add, %for.body ], [ %begin, %for.body.preheader ]
  %conv = trunc i64 %i.05 to i32
; CHECK: trunc i64 %{{[^ ]+}} to i32
  tail call void @_Z3usei(i32 %conv) #2
  %add = add nsw i64 %i.05, %stride
  %cmp = icmp eq i64 %add, %end
  br i1 %cmp, label %for.cond.cleanup.loopexit, label %for.body
}

declare void @_Z3usei(i32)

!nvvm.annotations = !{!0}
!0 = !{ptr @trunc_is_free, !"kernel", i32 1}
