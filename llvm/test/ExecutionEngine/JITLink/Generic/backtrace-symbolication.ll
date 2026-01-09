; RUN: rm -rf %t && mkdir -p %t
; RUN: llc -relocation-model=pic -filetype=obj -o %t/crash.o %s
; RUN: not --crash llvm-jitlink -debugger-support=false \
; RUN:     -write-symtab %t/crash.symtab.txt %t/crash.o \
; RUN:     > %t/backtrace.txt 2>&1
; RUN: llvm-jitlink -symbolicate-with %t/crash.symtab.txt %t/backtrace.txt \
; RUN:     | FileCheck %s

; Deliberately crash by dereferencing an environment variable that should never
; be defined, then symbolicate the backtrace using the dumped symbol table.

; UNSUPPORTED: system-windows

; CHECK: this_should_crash {{.*}} ({{.*}}crash.o)

@.str = private constant [52 x i8] c"a thousand curses upon anyone who dares define this\00", align 1

define i32 @this_should_crash() {
  %1 = call ptr @getenv(ptr noundef @.str)
  %2 = load i8, ptr %1, align 1
  %3 = sext i8 %2 to i32
  ret i32 %3
}

declare ptr @getenv(ptr)

define i32 @main(i32 %argc, ptr %argv) {
  %r = call i32 @this_should_crash()
  ret i32 %r
}
