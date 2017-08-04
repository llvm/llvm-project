; RUN: opt < %s -instcombine -S | FileCheck %s

%struct.C = type { %struct.C*, i32 }

; Check that we instcombine the load across the prefetch.

; CHECK-LABEL: define signext i32 @foo
define signext i32 @foo(%struct.C* %c) local_unnamed_addr #0 {
; CHECK: store i32 %dec, i32* %length_
; CHECK-NOT: load
; CHECK: llvm.prefetch
; CHECK-NEXT: ret
entry:
  %next_ = getelementptr inbounds %struct.C, %struct.C* %c, i32 0, i32 0
  %0 = load %struct.C*, %struct.C** %next_, align 8
  %next_1 = getelementptr inbounds %struct.C, %struct.C* %0, i32 0, i32 0
  %1 = load %struct.C*, %struct.C** %next_1, align 8
  store %struct.C* %1, %struct.C** %next_, align 8
  %length_ = getelementptr inbounds %struct.C, %struct.C* %c, i32 0, i32 1
  %2 = load i32, i32* %length_, align 8
  %dec = add nsw i32 %2, -1
  store i32 %dec, i32* %length_, align 8
  %3 = bitcast %struct.C* %1 to i8*
  call void @llvm.prefetch(i8* %3, i32 0, i32 0, i32 1)
  %4 = load i32, i32* %length_, align 8
  ret i32 %4
}

; Function Attrs: inaccessiblemem_or_argmemonly nounwind
declare void @llvm.prefetch(i8* nocapture readonly, i32, i32, i32) 

attributes #0 = { noinline nounwind }
; We've explicitly removed the function attrs from llvm.prefetch so we get the defaults.
; attributes #1 = { inaccessiblemem_or_argmemonly nounwind }
