; RUN: llc -mtriple i686-pc-windows-msvc < %s | FileCheck %s

; This test case is also intended to be run manually as a complete functional
; test. It should link, print something, and exit zero rather than crashing.
; It is the hypothetical lowering of a C source program that looks like:
;
;   int safe_div(int *n, int *d) {
;     int r;
;     __try {
;       __try {
;         r = *n / *d;
;       } __except(GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION) {
;         puts("EXCEPTION_ACCESS_VIOLATION");
;         r = -1;
;       }
;     } __except(GetExceptionCode() == EXCEPTION_INT_DIVIDE_BY_ZERO) {
;       puts("EXCEPTION_INT_DIVIDE_BY_ZERO");
;       r = -2;
;     }
;     return r;
;   }

@str1 = internal constant [27 x i8] c"EXCEPTION_ACCESS_VIOLATION\00"
@str2 = internal constant [29 x i8] c"EXCEPTION_INT_DIVIDE_BY_ZERO\00"

define i32 @safe_div(ptr %n, ptr %d) personality ptr @_except_handler3 {
entry:
  %r = alloca i32, align 4
  store i32 42, ptr %r
  invoke void @try_body(ptr %r, ptr %n, ptr %d)
          to label %__try.cont unwind label %lpad0

lpad0:
  %cs0 = catchswitch within none [label %handler0] unwind label %lpad1

handler0:
  %p0 = catchpad within %cs0 [ptr @safe_div_filt0]
  call void @puts(ptr @str1) [ "funclet"(token %p0) ]
  store i32 -1, ptr %r, align 4
  catchret from %p0 to label %__try.cont

lpad1:
  %cs1 = catchswitch within none [label %handler1] unwind to caller

handler1:
  %p1 = catchpad within %cs1 [ptr @safe_div_filt1]
  call void @puts(ptr @str2) [ "funclet"(token %p1) ]
  store i32 -2, ptr %r, align 4
  catchret from %p1 to label %__try.cont

__try.cont:
  %safe_ret = load i32, ptr %r, align 4
  ret i32 %safe_ret
}

; Normal path code

; CHECK: {{^}}_safe_div:
; CHECK: movl $42, [[rloc:.*\(%ebp\)]]
; CHECK: leal [[rloc]],
; CHECK: calll _try_body
; CHECK: [[cont_bb:LBB0_[0-9]+]]:
; CHECK: movl [[rloc]], %eax
; CHECK: retl

; Landing pad code

; CHECK: [[handler1:LBB0_[0-9]+]]: # %handler1
; 	Restore SP
; CHECK: movl {{.*}}(%ebp), %esp
; CHECK: calll _puts
; CHECK: jmp [[cont_bb]]

; CHECK: [[handler0:LBB0_[0-9]+]]: # %handler0
; 	Restore SP
; CHECK: movl {{.*}}(%ebp), %esp
; CHECK: calll _puts
; CHECK: jmp [[cont_bb]]

; CHECK: .section .xdata,"dr"
; CHECK: L__ehtable$safe_div:
; CHECK-NEXT: .long -1
; CHECK-NEXT: .long _safe_div_filt1
; CHECK-NEXT: .long [[handler1]]
; CHECK-NEXT: .long 0
; CHECK-NEXT: .long _safe_div_filt0
; CHECK-NEXT: .long [[handler0]]

define void @try_body(ptr %r, ptr %n, ptr %d) {
entry:
  %0 = load i32, ptr %n, align 4
  %1 = load i32, ptr %d, align 4
  %div = sdiv i32 %0, %1
  store i32 %div, ptr %r, align 4
  ret void
}

; The prototype of these filter functions is:
; int filter(EXCEPTION_POINTERS *eh_ptrs, ptr rbp);

; The definition of EXCEPTION_POINTERS is:
;   typedef struct _EXCEPTION_POINTERS {
;     EXCEPTION_RECORD *ExceptionRecord;
;     CONTEXT          *ContextRecord;
;   } EXCEPTION_POINTERS;

; The definition of EXCEPTION_RECORD is:
;   typedef struct _EXCEPTION_RECORD {
;     DWORD ExceptionCode;
;     ...
;   } EXCEPTION_RECORD;

define i32 @safe_div_filt0() {
  %ebp = call ptr @llvm.frameaddress(i32 1)
  %eh_ptrs.addr.i8 = getelementptr inbounds i8, ptr %ebp, i32 -20
  %eh_ptrs = load ptr, ptr %eh_ptrs.addr.i8
  %eh_rec = load ptr, ptr %eh_ptrs
  %eh_code = load i32, ptr %eh_rec
  ; EXCEPTION_ACCESS_VIOLATION = 0xC0000005
  %cmp = icmp eq i32 %eh_code, 3221225477
  %filt.res = zext i1 %cmp to i32
  ret i32 %filt.res
}
define i32 @safe_div_filt1() {
  %ebp = call ptr @llvm.frameaddress(i32 1)
  %eh_ptrs.addr.i8 = getelementptr inbounds i8, ptr %ebp, i32 -20
  %eh_ptrs = load ptr, ptr %eh_ptrs.addr.i8
  %eh_rec = load ptr, ptr %eh_ptrs
  %eh_code = load i32, ptr %eh_rec
  ; EXCEPTION_INT_DIVIDE_BY_ZERO = 0xC0000094
  %cmp = icmp eq i32 %eh_code, 3221225620
  %filt.res = zext i1 %cmp to i32
  ret i32 %filt.res
}

@str_result = internal constant [21 x i8] c"safe_div result: %d\0A\00"

define i32 @main() {
  %d.addr = alloca i32, align 4
  %n.addr = alloca i32, align 4

  store i32 10, ptr %n.addr, align 4
  store i32 2, ptr %d.addr, align 4
  %r1 = call i32 @safe_div(ptr %n.addr, ptr %d.addr)
  call void (ptr, ...) @printf(ptr @str_result, i32 %r1)

  store i32 10, ptr %n.addr, align 4
  store i32 0, ptr %d.addr, align 4
  %r2 = call i32 @safe_div(ptr %n.addr, ptr %d.addr)
  call void (ptr, ...) @printf(ptr @str_result, i32 %r2)

  %r3 = call i32 @safe_div(ptr %n.addr, ptr null)
  call void (ptr, ...) @printf(ptr @str_result, i32 %r3)
  ret i32 0
}

declare i32 @_except_handler3(...)
declare i32 @llvm.eh.typeid.for(ptr) readnone nounwind
declare void @puts(ptr)
declare void @printf(ptr, ...)
declare void @abort()
declare ptr @llvm.frameaddress(i32)
