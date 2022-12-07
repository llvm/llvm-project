; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: = memw{{.*}}circ

target triple = "hexagon"

@l = external global i32, align 4

; Function Attrs: nounwind optsize
define void @circ2() #0 {
entry:
  store i32 0, i32* @l, align 4
  %0 = tail call i8* @llvm.hexagon.circ.ldw(i8* undef, i8* undef, i32 150995968, i32 4)
  ret void
}

declare i8* @llvm.hexagon.circ.ldw(i8*, i8*, i32, i32) #1
attributes #0 = { nounwind optsize }
attributes #1 = { argmemonly nounwind }
