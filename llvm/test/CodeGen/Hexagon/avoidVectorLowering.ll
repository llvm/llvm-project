; RUN: llc -march=hexagon -O3 < %s | FileCheck %s
; CHECK-NOT: vmem

target triple = "hexagon-unknown--elf"

@g0 = common global [32 x i16] zeroinitializer, align 8

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  call void @llvm.memset.p0i8.i32(i8* align 8 bitcast ([32 x i16]* @g0 to i8*), i8 0, i32 64, i1 false)
  ret i32 0
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i1) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
attributes #1 = { argmemonly nounwind }
