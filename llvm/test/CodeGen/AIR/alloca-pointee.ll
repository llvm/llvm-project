; An alloca's typed pointer is to its allocated type: an access of another
; type (an i32 store into [16 x i32], an i8 GEP) goes through a pointer
; cast. Metal rejected such kernels ("Unexpected bitcode file!")
; when the writer took the alloca's pointee from its uses instead.
; RUN: env METAL_DUMP_PREWRITE=%t.ll llc -mtriple=air -filetype=obj -air-output=air %s -o /dev/null
; RUN: FileCheck %s < %t.ll

; CHECK-LABEL: define void @k(
; CHECK: %arr = alloca [16 x i32], align 4
; CHECK: %[[C0:.*]] = bitcast ptr %arr to ptr
; CHECK: store i32 %v, ptr %[[C0]], align 4
; CHECK: %[[C1:.*]] = bitcast ptr %arr to ptr
; CHECK: %g = getelementptr inbounds i8, ptr %[[C1]], i64 8
; CHECK: %[[C2:.*]] = bitcast ptr %g to ptr
; CHECK: store i32 %v, ptr %[[C2]], align 4
; CHECK: call void @llvm.memset.p0.i32(ptr align 4 %arr, i8 7, i32 %n, i1 false)
; CHECK: %e = getelementptr [16 x i32], ptr %arr, i32 0, i32 %v
define void @k(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %v = load i32, ptr addrspace(1) %in
  %n = and i32 %v, 63
  %arr = alloca [16 x i32], align 4
  store i32 %v, ptr %arr, align 4
  %g = getelementptr inbounds i8, ptr %arr, i64 8
  store i32 %v, ptr %g, align 4
  call void @llvm.memset.p0.i32(ptr align 4 %arr, i8 7, i32 %n, i1 false)
  %e = getelementptr [16 x i32], ptr %arr, i32 0, i32 %v
  %r = load i32, ptr %e, align 4
  store i32 %r, ptr addrspace(1) %out
  ret void
}

declare void @llvm.memset.p0.i32(ptr, i8, i32, i1 immarg)
