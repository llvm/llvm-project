; This check verifies that arguments passed by value get redzones.
; RUN: opt < %s -passes='asan-function-pipeline' -asan-realign-stack=32 -S | FileCheck %s
; RUN: opt < %s -passes='asan-function-pipeline' -asan-realign-stack=32 -asan-force-dynamic-shadow -S | FileCheck %s
; RUN: opt < %s -passes='asan-function-pipeline' -asan-realign-stack=32 -asan-mapping-scale=5 -S | FileCheck %s
; RUN: opt < %s -passes='asan-function-pipeline' -asan-realign-stack=32 -asan-force-dynamic-shadow -asan-mapping-scale=5 -S | FileCheck %s


target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { [8 x i32] }

declare i32 @bar(%struct.A*)

; Test behavior for named argument with explicit alignment.  The memcpy and
; alloca alignments should match the explicit alignment of 64.
define void @foo(%struct.A* byval(%struct.A) align 64 %a) sanitize_address {
entry:
; CHECK-LABEL: foo
; CHECK: call i64 @__asan_stack_malloc
; CHECK: alloca i8, i64 {{.*}} align 64
; CHECK: [[copyPtr:%[^ \t]+]] = inttoptr i64 %{{[^ \t]+}} to %struct.A*
; CHECK: [[copyBytePtr:%[^ \t]+]] = bitcast %struct.A* [[copyPtr]]
; CHECK: [[aBytePtr:%[^ \t]+]] = bitcast %struct.A* %a
; CHECK: call void @llvm.memcpy{{[^%]+}}[[copyBytePtr]]{{[^%]+}} align 64 [[aBytePtr]],{{[^,]+}},
; CHECK: call i32 @bar(%struct.A* [[copyPtr]])
; CHECK: ret void

  %call = call i32 @bar(%struct.A* %a)
  ret void
}

; Test behavior for unnamed argument without explicit alignment.  In this case,
; the first argument is referenced by the identifier %0 and the ABI requires a
; minimum alignment of 4 bytes since struct.A contains i32s which have 4-byte
; alignment.  However, the alloca alignment will be 32 since that is the value
; passed via the -asan-realign-stack option, which is greater than 4.
define void @baz(%struct.A* byval(%struct.A)) sanitize_address {
entry:
; CHECK-LABEL: baz
; CHECK: call i64 @__asan_stack_malloc
; CHECK: alloca i8, i64 {{.*}} align 32
; CHECK: [[copyPtr:%[^ \t]+]] = inttoptr i64 %{{[^ \t]+}} to %struct.A*
; CHECK: [[copyBytePtr:%[^ \t]+]] = bitcast %struct.A* [[copyPtr]]
; CHECK: [[aBytePtr:%[^ \t]+]] = bitcast %struct.A* %0
; CHECK: call void @llvm.memcpy{{[^%]+}}[[copyBytePtr]]{{[^%]+}} align 4 [[aBytePtr]],{{[^,]+}}
; CHECK: call i32 @bar(%struct.A* [[copyPtr]])
; CHECK: ret void

  %call = call i32 @bar(%struct.A* %0)
  ret void
}
