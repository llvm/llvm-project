; RUN: llc -march=mips < %s | FileCheck %s

%struct.sret0 = type { i32, i32, i32 }

define void @test0(%struct.sret0* noalias sret %agg.result, i32 %dummy) nounwind {
entry:
; CHECK: sw ${{[0-9]+}}, {{[0-9]+}}($4)
; CHECK: sw ${{[0-9]+}}, {{[0-9]+}}($4)
; CHECK: sw ${{[0-9]+}}, {{[0-9]+}}($4)
  getelementptr %struct.sret0* %agg.result, i32 0, i32 0    ; <i32*>:0 [#uses=1]
  store i32 %dummy, i32* %0, align 4
  getelementptr %struct.sret0* %agg.result, i32 0, i32 1    ; <i32*>:1 [#uses=1]
  store i32 %dummy, i32* %1, align 4
  getelementptr %struct.sret0* %agg.result, i32 0, i32 2    ; <i32*>:2 [#uses=1]
  store i32 %dummy, i32* %2, align 4
  ret void
}

