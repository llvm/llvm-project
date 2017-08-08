; RUN: llc -stack-symbol-ordering=0 < %s -mtriple=x86_64-linux | FileCheck %s

%struct.__va_list_tag = type { i32, i32, i8*, i8* }

define void @foo(i32 %x, ...) nounwind {
entry:
  %ap = alloca [1 x %struct.__va_list_tag], align 8; <[1 x %struct.__va_list_tag]*> [#uses=2]
  %ap12 = bitcast [1 x %struct.__va_list_tag]* %ap to i8*; <i8*> [#uses=2]
  call void @llvm.va_start(i8* %ap12)
; CHECK: testb %al, %al

; These test for specific offsets, which is very fragile. Still, the test needs
; to ensure that va_list has the correct element types.
;
; CHECK-DAG: movq {{.*}}, 192(%rsp)
; CHECK-DAG: movq {{.*}}, 184(%rsp)
; CHECK-DAG: movl {{.*}}, 180(%rsp)
; CHECK-DAG: movl {{.*}}, 176(%rsp)
  %ap3 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i64 0, i64 0; <%struct.__va_list_tag*> [#uses=1]
  call void @bar(%struct.__va_list_tag* %ap3) nounwind
  call void @llvm.va_end(i8* %ap12)
  ret void
}

declare void @llvm.va_start(i8*) nounwind

declare void @bar(%struct.__va_list_tag*)

declare void @llvm.va_end(i8*) nounwind
