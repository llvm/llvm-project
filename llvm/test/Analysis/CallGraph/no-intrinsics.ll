; RUN: opt < %s -passes=print-callgraph -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -passes=print-callgraph -disable-output 2>&1 | FileCheck %s

; Check that intrinsics aren't added to the call graph

declare void @llvm.memcpy.p0.p0.i32(ptr, ptr, i32, i1)

define void @f(ptr %out, ptr %in) {
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %out, ptr align 4 %in, i32 100, i1 false)
  ret void
}

; CHECK: Call graph node for function: 'f'
; CHECK-NOT: calls function 'llvm.memcpy.p0i8.p0i8.i32'
