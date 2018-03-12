; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that this testcase compiles successfully.
; Because l2fetch has mayLoad/mayStore flags on it, the packetizer
; was tricked into thinking that it's a store. The v65-specific
; code dealing with mem_shuff allowed it to be packetized together
; with the load.

; CHECK: l2fetch

target triple = "hexagon"

@g0 = external global [32768 x i8], align 8
@g1 = external local_unnamed_addr global [15 x i8*], align 8

; Function Attrs: nounwind
define void @f0() local_unnamed_addr #0 {
b0:
  store i8* inttoptr (i32 and (i32 sext (i8 ptrtoint (i8* getelementptr inbounds ([32768 x i8], [32768 x i8]* @g0, i32 0, i32 10000) to i8) to i32), i32 -65536) to i8*), i8** getelementptr inbounds ([15 x i8*], [15 x i8*]* @g1, i32 0, i32 1), align 4
  store i8* inttoptr (i32 and (i32 sext (i8 ptrtoint (i8* getelementptr inbounds ([32768 x i8], [32768 x i8]* @g0, i32 0, i32 10000) to i8) to i32), i32 -65536) to i8*), i8** getelementptr inbounds ([15 x i8*], [15 x i8*]* @g1, i32 0, i32 6), align 8
  tail call void @f1()
  %v0 = load i8*, i8** getelementptr inbounds ([15 x i8*], [15 x i8*]* @g1, i32 0, i32 0), align 8
  tail call void @llvm.hexagon.Y5.l2fetch(i8* %v0, i64 -9223372036854775808)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.hexagon.Y5.l2fetch(i8*, i64) #1

; Function Attrs: nounwind
declare void @f1() #1

attributes #0 = { nounwind "target-cpu"="hexagonv65" }
attributes #1 = { nounwind }
