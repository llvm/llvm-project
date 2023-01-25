; RUN: llc < %s 

; This testcase is invalid (the alignment specified for memcpy is 
; greater than the alignment guaranteed for Qux or C.0.1173), but it
; should compile, not crash the code generator.

@C.0.1173 = external constant [33 x i8]

define void @Bork() {
entry:
  %Qux = alloca [33 x i8]
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %Qux, ptr align 8 @C.0.1173, i64 33, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind
