; RUN: not llc -mtriple=bpf < %s 2> %t1
; RUN: FileCheck %s < %t1
; CHECK: by value not supported

%struct.S = type { [10 x i32] }

; Function Attrs: nounwind uwtable
define void @bar(i32 %a) #0 {
entry:
  %.compoundliteral = alloca %struct.S, align 8
  store i32 1, ptr %.compoundliteral, align 8
  %arrayinit.element = getelementptr inbounds %struct.S, ptr %.compoundliteral, i64 0, i32 0, i64 1
  store i32 2, ptr %arrayinit.element, align 4
  %arrayinit.element2 = getelementptr inbounds %struct.S, ptr %.compoundliteral, i64 0, i32 0, i64 2
  store i32 3, ptr %arrayinit.element2, align 8
  %arrayinit.start = getelementptr inbounds %struct.S, ptr %.compoundliteral, i64 0, i32 0, i64 3
  call void @llvm.memset.p0.i64(ptr align 4 %arrayinit.start, i8 0, i64 28, i1 false)
  call void @foo(i32 %a, ptr byval(%struct.S) align 8 %.compoundliteral) #3
  ret void
}

declare void @foo(i32, ptr byval(%struct.S) align 8) #1

; Function Attrs: nounwind
declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) #3
