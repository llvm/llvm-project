; RUN: llc < %s -march=x86 -o %t

%0 = type { i64, i64 }

declare fastcc %0 @ReturnBigStruct() nounwind readnone

define void @test(%0* %p) {
  %1 = call fastcc %0 @ReturnBigStruct()
  store %0 %1, %0* %p
  ret void
}

