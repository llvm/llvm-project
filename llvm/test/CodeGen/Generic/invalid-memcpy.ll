; RUN: llc < %s 

; This testcase is invalid (the alignment specified for memcpy is 
; greater than the alignment guaranteed for Qux or C.0.1173), but it
; should compile, not crash the code generator.

@C.0.1173 = external constant [33 x i8]

define void @Bork() {
entry:
  %Qux = alloca [33 x i8]
  %Qux1 = bitcast [33 x i8]* %Qux to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %Qux1, i8* getelementptr inbounds ([33 x i8], [33 x i8]* @C.0.1173, i32 0, i32 0), i64 33, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind
