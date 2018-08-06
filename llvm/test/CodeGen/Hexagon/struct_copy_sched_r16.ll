; RUN: llc -march=hexagon -hexagon-small-data-threshold=0 < %s | FileCheck %s
; Disable small-data, or the test will need to be modified to account for g0
; being placed there.

%s.3 = type { i8, i8, i8, i8, i8, i8 }

@g0 = external global %s.3

; CHECK: [[REG1:(r[0-9]+)]] = {{[#]+}}g0
; CHECK: r{{[0-9]+}} = mem{{u?}}b([[REG1]]+#{{[0-9]+}})
; CHECK: r0 = #0
; CHECK: dealloc_return
define i32 @f0() #0 {
b0:
  %v0 = alloca %s.3, align 1
  %v1 = getelementptr inbounds %s.3, %s.3* %v0, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 %v1, i8* align 1 getelementptr inbounds (%s.3, %s.3* @g0, i32 0, i32 0), i32 6, i1 false)
  call void @f1(%s.3* %v0) #0
  ret i32 0
}

declare void @f1(%s.3*)

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
