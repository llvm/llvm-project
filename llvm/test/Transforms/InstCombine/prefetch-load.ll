; RUN: opt < %s -passes=instcombine -S | FileCheck %s

%struct.C = type { ptr, i32 }

; Check that we instcombine the load across the prefetch.

; CHECK-LABEL: define signext i32 @foo
define signext i32 @foo(ptr %c) local_unnamed_addr #0 {
; CHECK: store i32 %dec, ptr %length_
; CHECK-NOT: load
; CHECK: llvm.prefetch
; CHECK-NEXT: ret
entry:
  %0 = load ptr, ptr %c, align 8
  %1 = load ptr, ptr %0, align 8
  store ptr %1, ptr %c, align 8
  %length_ = getelementptr inbounds %struct.C, ptr %c, i32 0, i32 1
  %2 = load i32, ptr %length_, align 8
  %dec = add nsw i32 %2, -1
  store i32 %dec, ptr %length_, align 8
  call void @llvm.prefetch(ptr %1, i32 0, i32 0, i32 1)
  %3 = load i32, ptr %length_, align 8
  ret i32 %3
}

; Function Attrs: inaccessiblemem_or_argmemonly nounwind
declare void @llvm.prefetch(ptr nocapture readonly, i32, i32, i32) 

attributes #0 = { noinline nounwind }
; We've explicitly removed the function attrs from llvm.prefetch so we get the defaults.
; attributes #1 = { inaccessiblemem_or_argmemonly nounwind }
