; RUN: opt < %s -passes=instcombine -S | FileCheck %s
;
; Check that SimplifyLibCalls do not (crash or) emit a library call if user
; has made a function alias with the same name.

%struct._IO_FILE = type { i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, [1 x i8], ptr, i64, ptr, ptr, ptr, ptr, i64, i32, [20 x i8] }
%struct._IO_marker = type { ptr, ptr, i32 }

@stderr = external global ptr, align 8
@.str = private constant [8 x i8] c"crash!\0A\00", align 1

@fwrite = alias i64 (ptr, i64, i64, ptr), ptr @__fwrite_alias

define i64 @__fwrite_alias(ptr %ptr, i64 %size, i64 %n, ptr %s) {
; CHECK-LABEL: @__fwrite_alias(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i64 0
;
entry:
  %ptr.addr = alloca ptr, align 8
  %size.addr = alloca i64, align 8
  %n.addr = alloca i64, align 8
  %s.addr = alloca ptr, align 8
  store ptr %ptr, ptr %ptr.addr, align 8
  store i64 %size, ptr %size.addr, align 8
  store i64 %n, ptr %n.addr, align 8
  store ptr %s, ptr %s.addr, align 8
  ret i64 0
}

define void @foo() {
; CHECK-LABEL: @foo(
; CHECK-NOT:    call i64 @fwrite(
; CHECK:        call {{.*}} @fprintf(
;
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  %0 = load ptr, ptr @stderr, align 8
  %call = call i32 (ptr, ptr, ...) @fprintf(ptr %0, ptr @.str)
  ret void
}

declare i32 @fprintf(ptr, ptr, ...)
