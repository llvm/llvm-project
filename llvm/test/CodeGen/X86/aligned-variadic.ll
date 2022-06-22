; RUN: llc < %s -mtriple=x86_64-apple-darwin -stack-symbol-ordering=0 | FileCheck %s -check-prefix=X64
; RUN: llc < %s -mtriple=i686-apple-darwin -stack-symbol-ordering=0 | FileCheck %s -check-prefix=X32

%struct.Baz = type { [17 x i8] }
%struct.__va_list_tag = type { i32, i32, ptr, ptr }

; Function Attrs: nounwind uwtable
define void @bar(ptr byval(%struct.Baz) nocapture readnone align 8 %x, ...) {
entry:
  %va = alloca [1 x %struct.__va_list_tag], align 16
  call void @llvm.va_start(ptr %va)
  %overflow_arg_area_p = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %va, i64 0, i64 0, i32 2
  %overflow_arg_area = load ptr, ptr %overflow_arg_area_p, align 8
  %overflow_arg_area.next = getelementptr i8, ptr %overflow_arg_area, i64 24
  store ptr %overflow_arg_area.next, ptr %overflow_arg_area_p, align 8
; X32: leal    68(%esp), [[REG:%.*]]
; X32: movl    [[REG]], 16(%esp)
; X64: leaq    256(%rsp), [[REG:%.*]]
; X64: movq    [[REG]], 184(%rsp)
; X64: leaq    176(%rsp), %rdi
  call void @qux(ptr %va)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.va_start(ptr)

declare void @qux(ptr)
