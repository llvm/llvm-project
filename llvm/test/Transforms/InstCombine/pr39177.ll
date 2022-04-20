; RUN: opt < %s -passes=instcombine -S | FileCheck %s
;
; Check that SimplifyLibCalls do not (crash or) emit a library call if user
; has made a function alias with the same name.

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@stderr = external global %struct._IO_FILE*, align 8
@.str = private constant [8 x i8] c"crash!\0A\00", align 1

@fwrite = alias i64 (i8*, i64, i64, %struct._IO_FILE*), i64 (i8*, i64, i64, %struct._IO_FILE*)* @__fwrite_alias

define i64 @__fwrite_alias(i8* %ptr, i64 %size, i64 %n, %struct._IO_FILE* %s) {
; CHECK-LABEL: @__fwrite_alias(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i64 0
;
entry:
  %ptr.addr = alloca i8*, align 8
  %size.addr = alloca i64, align 8
  %n.addr = alloca i64, align 8
  %s.addr = alloca %struct._IO_FILE*, align 8
  store i8* %ptr, i8** %ptr.addr, align 8
  store i64 %size, i64* %size.addr, align 8
  store i64 %n, i64* %n.addr, align 8
  store %struct._IO_FILE* %s, %struct._IO_FILE** %s.addr, align 8
  ret i64 0
}

define void @foo() {
; CHECK-LABEL: @foo(
; CHECK-NOT:    call i64 @fwrite(
; CHECK:        call {{.*}} @fprintf(
;
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %0 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %call = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %0, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str, i32 0, i32 0))
  ret void
}

declare i32 @fprintf(%struct._IO_FILE*, i8*, ...)
