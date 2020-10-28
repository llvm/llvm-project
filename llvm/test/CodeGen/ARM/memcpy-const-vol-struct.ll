; RUN: llc -mtriple=armv7-arm-none-eabi -o - %s | FileCheck %s

%struct.sMyType = type { i32 }

@val = hidden constant %struct.sMyType zeroinitializer, align 4
@v = internal global %struct.sMyType zeroinitializer, align 4

define hidden void @InitVal() local_unnamed_addr {
entry:
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 bitcast (%struct.sMyType* @v to i8*), i8* align 4 bitcast (%struct.sMyType* @val to i8*), i32 4, i1 true)
; The last argument is the isvolatile argument. This is a volatile memcpy.
; Test that the memcpy expansion does not optimize away the load.
; CHECK: ldr
; CHECK: str
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i32, i1 immarg)
