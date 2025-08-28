; RUN: llc < %s -O0 -march=nvptx -mattr=+ptx60 -mcpu=sm_30 | FileCheck %s --check-prefixes=CHECK,CHECK32
; RUN: llc < %s -O0 -march=nvptx64 -mattr=+ptx60 -mcpu=sm_30 | FileCheck %s --check-prefixes=CHECK,CHECK64
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -O0 -march=nvptx -mattr=+ptx60 -mcpu=sm_30 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -O0 -march=nvptx64 -mattr=+ptx60 -mcpu=sm_30 | %ptxas-verify %}

; CHECK: .address_size [[BITS:32|64]]

%struct.__va_list_tag = type { ptr, ptr, i32, i32 }

@foo_ptr = internal addrspace(1) global ptr @foo, align 8

define i32 @foo(i32 %a, ...) {
entry:
  %al = alloca [1 x %struct.__va_list_tag], align 8
  %al2 = alloca [1 x %struct.__va_list_tag], align 8

; Test va_start
; CHECK:         .param .align 8 .b8 foo_vararg[]
; CHECK:         mov.b[[BITS]] [[VA_PTR:%(r|rd)[0-9]+]], foo_vararg;
; CHECK-NEXT:    st.b[[BITS]] [%SP], [[VA_PTR]];

  call void @llvm.va_start(ptr %al)

; Test va_copy()
; CHECK-NEXT:	 ld.b[[BITS]] [[VA_PTR:%(r|rd)[0-9]+]], [%SP];
; CHECK-NEXT:	 st.b[[BITS]] [%SP+{{[0-9]+}}], [[VA_PTR]];

  call void @llvm.va_copy(ptr %al2, ptr %al)

; Test va_arg(ap, int32_t)
; CHECK-NEXT:    ld.b[[BITS]] [[VA_PTR:%(r|rd)[0-9]+]], [%SP];
; CHECK-NEXT:    add.s[[BITS]] [[VA_PTR_TMP:%(r|rd)[0-9]+]], [[VA_PTR]], 3;
; CHECK-NEXT:    and.b[[BITS]] [[VA_PTR_ALIGN:%(r|rd)[0-9]+]], [[VA_PTR_TMP]], -4;
; CHECK-NEXT:    add.s[[BITS]] [[VA_PTR_NEXT:%(r|rd)[0-9]+]], [[VA_PTR_ALIGN]], 4;
; CHECK-NEXT:    st.b[[BITS]] [%SP], [[VA_PTR_NEXT]];
; CHECK-NEXT:    ld.local.b32 %r{{[0-9]+}}, [[[VA_PTR_ALIGN]]];

  %0 = va_arg ptr %al, i32

; Test va_arg(ap, int64_t)
; CHECK-NEXT:    ld.b[[BITS]] [[VA_PTR:%(r|rd)[0-9]+]], [%SP];
; CHECK-NEXT:    add.s[[BITS]] [[VA_PTR_TMP:%(r|rd)[0-9]+]], [[VA_PTR]], 7;
; CHECK-NEXT:    and.b[[BITS]] [[VA_PTR_ALIGN:%(r|rd)[0-9]+]], [[VA_PTR_TMP]], -8;
; CHECK-NEXT:    add.s[[BITS]] [[VA_PTR_NEXT:%(r|rd)[0-9]+]], [[VA_PTR_ALIGN]], 8;
; CHECK-NEXT:    st.b[[BITS]] [%SP], [[VA_PTR_NEXT]];
; CHECK-NEXT:    ld.local.b64 %rd{{[0-9]+}}, [[[VA_PTR_ALIGN]]];

  %1 = va_arg ptr %al, i64

; Test va_arg(ap, double)
; CHECK-NEXT:    ld.b[[BITS]] [[VA_PTR:%(r|rd)[0-9]+]], [%SP];
; CHECK-NEXT:    add.s[[BITS]] [[VA_PTR_TMP:%(r|rd)[0-9]+]], [[VA_PTR]], 7;
; CHECK-NEXT:    and.b[[BITS]] [[VA_PTR_ALIGN:%(r|rd)[0-9]+]], [[VA_PTR_TMP]], -8;
; CHECK-NEXT:    add.s[[BITS]] [[VA_PTR_NEXT:%(r|rd)[0-9]+]], [[VA_PTR_ALIGN]], 8;
; CHECK-NEXT:    st.b[[BITS]] [%SP], [[VA_PTR_NEXT]];
; CHECK-NEXT:    ld.local.b64 %rd{{[0-9]+}}, [[[VA_PTR_ALIGN]]];

  %2 = va_arg ptr %al, double

; Test va_arg(ap, ptr)
; CHECK-NEXT:    ld.b[[BITS]] [[VA_PTR:%(r|rd)[0-9]+]], [%SP];
; CHECK32-NEXT:  add.s32 [[VA_PTR_TMP:%r[0-9]+]], [[VA_PTR]], 3;
; CHECK64-NEXT:  add.s64 [[VA_PTR_TMP:%rd[0-9]+]], [[VA_PTR]], 7;
; CHECK32-NEXT:  and.b32 [[VA_PTR_ALIGN:%r[0-9]+]], [[VA_PTR_TMP]], -4;
; CHECK64-NEXT:  and.b64 [[VA_PTR_ALIGN:%rd[0-9]+]], [[VA_PTR_TMP]], -8;
; CHECK32-NEXT:  add.s32 [[VA_PTR_NEXT:%r[0-9]+]], [[VA_PTR_ALIGN]], 4;
; CHECK64-NEXT:  add.s64 [[VA_PTR_NEXT:%rd[0-9]+]], [[VA_PTR_ALIGN]], 8;
; CHECK-NEXT:    st.b[[BITS]] [%SP], [[VA_PTR_NEXT]];
; CHECK-NEXT:    ld.local.b[[BITS]] %{{(r|rd)[0-9]+}}, [[[VA_PTR_ALIGN]]];

  %3 = va_arg ptr %al, ptr
  %call = call i32 @bar(i32 %a, i32 %0, i64 %1, double %2, ptr %3)

  call void @llvm.va_end(ptr %al)
  %4 =  va_arg ptr %al2, i32
  call void @llvm.va_end(ptr %al2)
  %5 = add i32 %call, %4
  ret i32 %5
}

define i32 @test_foo(i32 %i, i64 %l, double %d, ptr %p) {
; Test indirect variadic function call.

; Load arguments to temporary variables
; CHECK32:       ld.param.b32 [[ARG_VOID_PTR:%r[0-9]+]], [test_foo_param_3];
; CHECK64:       ld.param.b64 [[ARG_VOID_PTR:%rd[0-9]+]], [test_foo_param_3];
; CHECK-NEXT:    ld.param.b64 [[ARG_DOUBLE:%rd[0-9]+]], [test_foo_param_2];
; CHECK-NEXT:    ld.param.b64 [[ARG_I64:%rd[0-9]+]], [test_foo_param_1];
; CHECK-NEXT:    ld.param.b32 [[ARG_I32:%r[0-9]+]], [test_foo_param_0];

; Store arguments to an array
; CHECK32:      .param .align 8 .b8 param1[28];
; CHECK64:      .param .align 8 .b8 param1[32];
; CHECK-DAG:    .param .b32 retval0;
; CHECK-DAG:    st.param.b32 [param1], [[ARG_I32]];
; CHECK-DAG:    st.param.b64 [param1+8], [[ARG_I64]];
; CHECK-DAG:    st.param.b64 [param1+16], [[ARG_DOUBLE]];
; CHECK-DAG:    st.param.b[[BITS]] [param1+24], [[ARG_VOID_PTR]];
; CHECK-DAG:    prototype_1 : .callprototype (.param .b32 _) _ (.param .b32 _, .param .align 8 .b8 _[]

entry:
  %ptr = load ptr, ptr addrspacecast (ptr addrspace(1) @foo_ptr to ptr), align 8
  %call = call i32 (i32, ...) %ptr(i32 4, i32 %i, i64 %l, double %d, ptr %p)
  ret i32 %call
}

declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)
declare void @llvm.va_copy(ptr, ptr)
declare i32 @bar(i32, i32, i64, double, ptr)
