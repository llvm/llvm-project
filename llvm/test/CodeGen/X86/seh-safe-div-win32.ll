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

define i32 @safe_div(i32* %n, i32* %d) personality i8* bitcast (i32 (...)* @_except_handler3 to i8*) {
entry:
  %r = alloca i32, align 4
  store i32 42, i32* %r
  invoke void @try_body(i32* %r, i32* %n, i32* %d)
          to label %__try.cont unwind label %lpad0

lpad0:
  %p0 = catchpad [i8* bitcast (i32 ()* @safe_div_filt0 to i8*)]
          to label %handler0 unwind label %endpad0

handler0:
  call void @puts(i8* getelementptr ([27 x i8], [27 x i8]* @str1, i32 0, i32 0))
  store i32 -1, i32* %r, align 4
  catchret %p0 to label %__try.cont

endpad0:
  catchendpad unwind label %lpad1

lpad1:
  %p1 = catchpad [i8* bitcast (i32 ()* @safe_div_filt1 to i8*)]
          to label %handler1 unwind label %endpad1

handler1:
  call void @puts(i8* getelementptr ([29 x i8], [29 x i8]* @str2, i32 0, i32 0))
  store i32 -2, i32* %r, align 4
  catchret %p1 to label %__try.cont

endpad1:
  catchendpad unwind to caller

__try.cont:
  %safe_ret = load i32, i32* %r, align 4
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

; CHECK: [[lpad0:LBB0_[0-9]+]]: # %lpad0
; 	Restore SP
; CHECK: movl {{.*}}(%ebp), %esp
; CHECK: calll _puts
; CHECK: jmp [[cont_bb]]

; CHECK: [[lpad1:LBB0_[0-9]+]]: # %lpad1
; 	Restore SP
; CHECK: movl {{.*}}(%ebp), %esp
; CHECK: calll _puts
; CHECK: jmp [[cont_bb]]

; CHECK: .section .xdata,"dr"
; CHECK: L__ehtable$safe_div:
; CHECK-NEXT: .long -1
; CHECK-NEXT: .long _safe_div_filt1
; CHECK-NEXT: .long [[lpad1]]
; CHECK-NEXT: .long 0
; CHECK-NEXT: .long _safe_div_filt0
; CHECK-NEXT: .long [[lpad0]]

define void @try_body(i32* %r, i32* %n, i32* %d) {
entry:
  %0 = load i32, i32* %n, align 4
  %1 = load i32, i32* %d, align 4
  %div = sdiv i32 %0, %1
  store i32 %div, i32* %r, align 4
  ret void
}

; The prototype of these filter functions is:
; int filter(EXCEPTION_POINTERS *eh_ptrs, void *rbp);

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
  %ebp = call i8* @llvm.frameaddress(i32 1)
  %eh_ptrs.addr.i8 = getelementptr inbounds i8, i8* %ebp, i32 -20
  %eh_ptrs.addr = bitcast i8* %eh_ptrs.addr.i8 to i32***
  %eh_ptrs = load i32**, i32*** %eh_ptrs.addr
  %eh_rec = load i32*, i32** %eh_ptrs
  %eh_code = load i32, i32* %eh_rec
  ; EXCEPTION_ACCESS_VIOLATION = 0xC0000005
  %cmp = icmp eq i32 %eh_code, 3221225477
  %filt.res = zext i1 %cmp to i32
  ret i32 %filt.res
}
define i32 @safe_div_filt1() {
  %ebp = call i8* @llvm.frameaddress(i32 1)
  %eh_ptrs.addr.i8 = getelementptr inbounds i8, i8* %ebp, i32 -20
  %eh_ptrs.addr = bitcast i8* %eh_ptrs.addr.i8 to i32***
  %eh_ptrs = load i32**, i32*** %eh_ptrs.addr
  %eh_rec = load i32*, i32** %eh_ptrs
  %eh_code = load i32, i32* %eh_rec
  ; EXCEPTION_INT_DIVIDE_BY_ZERO = 0xC0000094
  %cmp = icmp eq i32 %eh_code, 3221225620
  %filt.res = zext i1 %cmp to i32
  ret i32 %filt.res
}

@str_result = internal constant [21 x i8] c"safe_div result: %d\0A\00"

define i32 @main() {
  %d.addr = alloca i32, align 4
  %n.addr = alloca i32, align 4

  store i32 10, i32* %n.addr, align 4
  store i32 2, i32* %d.addr, align 4
  %r1 = call i32 @safe_div(i32* %n.addr, i32* %d.addr)
  call void (i8*, ...) @printf(i8* getelementptr ([21 x i8], [21 x i8]* @str_result, i32 0, i32 0), i32 %r1)

  store i32 10, i32* %n.addr, align 4
  store i32 0, i32* %d.addr, align 4
  %r2 = call i32 @safe_div(i32* %n.addr, i32* %d.addr)
  call void (i8*, ...) @printf(i8* getelementptr ([21 x i8], [21 x i8]* @str_result, i32 0, i32 0), i32 %r2)

  %r3 = call i32 @safe_div(i32* %n.addr, i32* null)
  call void (i8*, ...) @printf(i8* getelementptr ([21 x i8], [21 x i8]* @str_result, i32 0, i32 0), i32 %r3)
  ret i32 0
}

declare i32 @_except_handler3(...)
declare i32 @llvm.eh.typeid.for(i8*) readnone nounwind
declare void @puts(i8*)
declare void @printf(i8*, ...)
declare void @abort()
declare i8* @llvm.frameaddress(i32)
