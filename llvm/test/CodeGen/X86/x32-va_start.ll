; RUN: llc < %s -mtriple=x86_64-linux-gnux32 | FileCheck %s -check-prefix=CHECK -check-prefix=SSE
; RUN: llc < %s -mtriple=x86_64-linux-gnux32 -mattr=-sse | FileCheck %s -check-prefix=CHECK -check-prefix=NOSSE
;
; Verifies that x32 va_start lowering is sane. To regenerate this test, use
; cat <<EOF |
; #include <stdarg.h>
;
; int foo(float a, const char* fmt, ...) {
;   va_list ap;
;   va_start(ap, fmt);
;   int value = va_arg(ap, int);
;   va_end(ap);
;   return value;
; }
; EOF
; build/bin/clang -mx32 -O3 -o- -S -emit-llvm -xc -
;
target datalayout = "e-m:e-p:32:32-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnux32"

%struct.__va_list_tag = type { i32, i32, i8*, i8* }

define i32 @foo(float %a, i8* nocapture readnone %fmt, ...) nounwind {
entry:
  %ap = alloca [1 x %struct.__va_list_tag], align 16
  %0 = bitcast [1 x %struct.__va_list_tag]* %ap to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %0) #2
  call void @llvm.va_start(i8* %0)
; SSE: subl $72, %esp
; SSE: testb %al, %al
; SSE: je .[[NOFP:.*]]
; SSE-DAG: movaps %xmm1
; SSE-DAG: movaps %xmm2
; SSE-DAG: movaps %xmm3
; SSE-DAG: movaps %xmm4
; SSE-DAG: movaps %xmm5
; SSE-DAG: movaps %xmm6
; SSE-DAG: movaps %xmm7
; NOSSE-NOT: xmm
; SSE: .[[NOFP]]:
; CHECK-DAG: movq %r9
; CHECK-DAG: movq %r8
; CHECK-DAG: movq %rcx
; CHECK-DAG: movq %rdx
; CHECK-DAG: movq %rsi
  %gp_offset_p = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0, i32 0
  %gp_offset = load i32, i32* %gp_offset_p, align 16
  %fits_in_gp = icmp ult i32 %gp_offset, 41
  br i1 %fits_in_gp, label %vaarg.in_reg, label %vaarg.in_mem
; CHECK: cmpl $40, [[COUNT:.*]]
; CHECK: ja .[[IN_MEM:.*]]

vaarg.in_reg:                                     ; preds = %entry
  %1 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0, i32 3
  %reg_save_area = load i8*, i8** %1, align 4
  %2 = getelementptr i8, i8* %reg_save_area, i32 %gp_offset
  %3 = add i32 %gp_offset, 8
  store i32 %3, i32* %gp_offset_p, align 16
  br label %vaarg.end
; CHECK: movl {{[^,]*}}, [[ADDR:.*]]
; CHECK: addl [[COUNT]], [[ADDR]]
; SSE: jmp .[[END:.*]]
; NOSSE: movl ([[ADDR]]), %eax
; NOSSE: retq
; CHECK: .[[IN_MEM]]:
vaarg.in_mem:                                     ; preds = %entry
  %overflow_arg_area_p = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0, i32 2
  %overflow_arg_area = load i8*, i8** %overflow_arg_area_p, align 8
  %overflow_arg_area.next = getelementptr i8, i8* %overflow_arg_area, i32 8
  store i8* %overflow_arg_area.next, i8** %overflow_arg_area_p, align 8
  br label %vaarg.end
; CHECK: movl {{[^,]*}}, [[ADDR]]
; NOSSE: movl ([[ADDR]]), %eax
; NOSSE: retq
; SSE: .[[END]]:

vaarg.end:                                        ; preds = %vaarg.in_mem, %vaarg.in_reg
  %vaarg.addr.in = phi i8* [ %2, %vaarg.in_reg ], [ %overflow_arg_area, %vaarg.in_mem ]
  %vaarg.addr = bitcast i8* %vaarg.addr.in to i32*
  %4 = load i32, i32* %vaarg.addr, align 4
  call void @llvm.va_end(i8* %0)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %0) #2
  ret i32 %4
; SSE: movl ([[ADDR]]), %eax
; SSE: retq
}

; Function Attrs: nounwind argmemonly
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) nounwind

; Function Attrs: nounwind
declare void @llvm.va_start(i8*) nounwind

; Function Attrs: nounwind
declare void @llvm.va_end(i8*) nounwind

; Function Attrs: nounwind argmemonly
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) nounwind

