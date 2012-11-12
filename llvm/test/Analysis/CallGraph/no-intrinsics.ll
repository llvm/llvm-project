; RUN: opt < %s -print-callgraph -disable-output 2>&1 | FileCheck %s

; Check that intrinsics aren't added to the call graph

declare void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i32, i1)

define void @f(i8* %out, i8* %in) {
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %out, i8* %in, i32 100, i32 4, i1 false)
  ret void
}

; CHECK: Call graph node for function: 'f'
; CHECK-NOT: calls function 'llvm.memcpy.p0i8.p0i8.i32'