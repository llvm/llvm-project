; RUN: opt < %s -memcpyopt -S | FileCheck %s

%struct.MV = type { i16, i16 }

define void @test(ptr nocapture %c) nounwind optsize {
; All the stores in this example should be merged into a single memset.
; CHECK-NOT:  store i32 -1
; CHECK: call void @llvm.memset.p0.i64
  store i32 -1, ptr %c, align 4
  %1 = getelementptr inbounds i32, ptr %c, i32 1
  store i32 -1, ptr %1, align 4
  %2 = getelementptr inbounds i32, ptr %c, i32 2
  store i32 -1, ptr %2, align 4
  call void @llvm.pseudoprobe(i64 5116412291814990879, i64 1, i32 0, i64 -1)
  %3 = getelementptr inbounds i32, ptr %c, i32 3
  store i32 -1, ptr %3, align 4
  %4 = getelementptr inbounds i32, ptr %c, i32 4
  store i32 -1, ptr %4, align 4
  ret void
}

; Function Attrs: inaccessiblememonly nounwind willreturn
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #0

attributes #0 = { inaccessiblememonly nounwind willreturn }
