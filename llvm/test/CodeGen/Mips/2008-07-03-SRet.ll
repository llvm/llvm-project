; RUN: llc -mtriple=mips < %s | FileCheck %s

%struct.sret0 = type { i32, i32, i32 }

define void @test0(ptr noalias sret(%struct.sret0) %agg.result, i32 %dummy) nounwind {
entry:
; CHECK: sw ${{[0-9]+}}, {{[0-9]+}}($4)
; CHECK: sw ${{[0-9]+}}, {{[0-9]+}}($4)
; CHECK: sw ${{[0-9]+}}, {{[0-9]+}}($4)
  getelementptr %struct.sret0, ptr %agg.result, i32 0, i32 0    ; <ptr>:0 [#uses=1]
  store i32 %dummy, ptr %0, align 4
  getelementptr %struct.sret0, ptr %agg.result, i32 0, i32 1    ; <ptr>:1 [#uses=1]
  store i32 %dummy, ptr %1, align 4
  getelementptr %struct.sret0, ptr %agg.result, i32 0, i32 2    ; <ptr>:2 [#uses=1]
  store i32 %dummy, ptr %2, align 4
  ret void
}

