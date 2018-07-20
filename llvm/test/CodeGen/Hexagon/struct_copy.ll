; RUN: llc -march=hexagon -hexagon-small-data-threshold=0 < %s | FileCheck %s
; Disable small-data, or otherwise g3 will end up in .sdata. While that is
; not a problem, this test was originally written with the g3 not being in
; there, so keep it that way.

%s.0 = type { i32, i32, i32, i32, i32, i32 }
%s.1 = type { i64, i64, i64, i64, i64, i64 }
%s.2 = type { i16, i16, i16, i16, i16, i16 }
%s.3 = type { i8, i8, i8, i8, i8, i8 }

@g0 = external global %s.0
@g1 = external global %s.1
@g2 = external global %s.2
@g3 = external global %s.3

; CHECK-LABEL: f0:
; CHECK: [[REG1:(r[0-9]+)]] = {{[#]+}}g0
; CHECK: r{{[0-9]+}} = memw([[REG1]]+#{{[0-9]+}})
; CHECK-NOT: = memd
; CHECK: dealloc_return
define i32 @f0() #0 {
b0:
  %v0 = alloca %s.0, align 4
  %v1 = bitcast %s.0* %v0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %v1, i8* align 4 bitcast (%s.0* @g0 to i8*), i32 24, i1 false)
  call void @f1(%s.0* %v0) #0
  ret i32 0
}

declare void @f1(%s.0*)

; CHECK-LABEL: f2:
; CHECK: [[REG2:(r[0-9]+)]] = {{[#]+}}g1
; CHECK: r{{[0-9]+}}:{{[0-9]+}} = memd([[REG2]]+#{{[0-9]+}})
; CHECK: dealloc_return
define i32 @f2() #0 {
b0:
  %v0 = alloca %s.1, align 8
  %v1 = bitcast %s.1* %v0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %v1, i8* align 8 bitcast (%s.1* @g1 to i8*), i32 48, i1 false)
  call void @f3(%s.1* %v0) #0
  ret i32 0
}

declare void @f3(%s.1*)

; CHECK-LABEL: f4:
; CHECK: [[REG1:(r[0-9]+)]] = {{[#]+}}g2
; CHECK: r{{[0-9]+}} = mem{{u?}}h([[REG1]]+#{{[0-9]+}})
; CHECK-NOT: = memd
; CHECK: dealloc_return
define i32 @f4() #0 {
b0:
  %v0 = alloca %s.2, align 2
  %v1 = bitcast %s.2* %v0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 %v1, i8* align 2 bitcast (%s.2* @g2 to i8*), i32 12, i1 false)
  call void @f5(%s.2* %v0) #0
  ret i32 0
}

declare void @f5(%s.2*)

; CHECK-LABEL: f6:
; CHECK: [[REG1:(r[0-9]+)]] = {{[#]+}}g3
; CHECK: r{{[0-9]+}} = mem{{u?}}b([[REG1]]+#{{[0-9]+}})
; CHECK-NOT: = memw
; CHECK: dealloc_return
define i32 @f6() #0 {
b0:
  %v0 = alloca %s.3, align 1
  %v1 = getelementptr inbounds %s.3, %s.3* %v0, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 %v1, i8* align 1 getelementptr inbounds (%s.3, %s.3* @g3, i32 0, i32 0), i32 6, i1 false)
  call void @f7(%s.3* %v0) #0
  ret i32 0
}

declare void @f7(%s.3*)

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
