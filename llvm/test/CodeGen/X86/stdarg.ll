; RUN: llc -stack-symbol-ordering=0 < %s -mtriple=x86_64-linux | FileCheck %s

%struct.__va_list_tag = type { i32, i32, ptr, ptr }

define void @foo(i32 %x, ...) nounwind {
entry:
  %ap = alloca [1 x %struct.__va_list_tag], align 8; <ptr> [#uses=2]
  call void @llvm.va_start(ptr %ap)
; CHECK: testb %al, %al

; These test for specific offsets, which is very fragile. Still, the test needs
; to ensure that va_list has the correct element types.
;
; CHECK-DAG: movq {{.*}}, 192(%rsp)
; CHECK-DAG: movq {{.*}}, 184(%rsp)
; CHECK-DAG: movq {{.*}}, 176(%rsp)
  %ap3 = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %ap, i64 0, i64 0; <ptr> [#uses=1]
  call void @bar(ptr %ap3) nounwind
  call void @llvm.va_end(ptr %ap)
  ret void
}

declare void @llvm.va_start(ptr) nounwind

declare void @bar(ptr)

declare void @llvm.va_end(ptr) nounwind
