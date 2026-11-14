; RUN: opt -mtriple=bpfel -passes=bpf-arena-data -S < %s | FileCheck %s
; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+arena -filetype=obj -o %t < %s
; RUN: llvm-readelf -S -r %t | FileCheck --check-prefix=ELF %s
; RUN: not opt -mtriple=bpfel -passes=bpf-arena-data -disable-output %S/Inputs/arena-data-err.ll 2>&1 \
; RUN:   | FileCheck --check-prefix=ERR %s

; Global data is in arena.

; Variables, constants and strings that are not zero are in one section.
; CHECK-DAG: @var = global i64 5, section ".addr_space.1"
; CHECK-DAG: @table = internal global [2 x i64] [i64 7, i64 8], section ".addr_space.1"
; CHECK-DAG: @str = private global [6 x i8] c"hello\00", section ".addr_space.1", align 1
@var = global i64 5
@table = internal constant [2 x i64] [i64 7, i64 8]
@str = private unnamed_addr constant [6 x i8] c"hello\00", align 1

; Those that are zero take no space in the file.
; CHECK-DAG: @zero = global [1024 x i64] zeroinitializer, section ".bss.addr_space.1"
; CHECK-DAG: @uninit = internal global ptr poison, section ".bss.addr_space.1"
@zero = global [1024 x i64] zeroinitializer
@uninit = internal global ptr poison

; The loader resolves pointers.
; CHECK-DAG: @ptr = global ptr getelementptr {{.*}}(i8, ptr @zero, i64 8), section ".addr_space.1"
@ptr = global ptr getelementptr (i8, ptr @zero, i64 8)

; Only there, so a constant that has one is in arena too.
; CHECK-DAG: @const_ptr = global ptr @zero, section ".addr_space.1"
@const_ptr = constant ptr @zero

; The loader sets a constant that is not static, the kernel knows its value.
; CHECK-DAG: @config = constant i32 1{{$}}
@config = constant i32 1

; There is no pointer to a constant that is only loaded from.
; CHECK-DAG: @direct = internal constant [2 x i64] [i64 1, i64 2]{{$}}
@direct = internal constant [2 x i64] [i64 1, i64 2]

; What has a section stays there.
; CHECK-DAG: @_license = global [4 x i8] c"GPL\00", section "license"
@_license = global [4 x i8] c"GPL\00", section "license"

; The kernel looks for pointers to functions in read-only data.
; CHECK-DAG: @fns = internal constant [1 x ptr] {{\[}}ptr @fn]{{$}}
@fns = internal constant [1 x ptr] [ptr @fn]

; A kfunc takes the string from a read-only map.
; CHECK-DAG: @fmt = private unnamed_addr constant [3 x i8] c"%d\00", align 1{{$}}
@fmt = private unnamed_addr constant [3 x i8] c"%d\00", align 1

; It does, the program has a pointer to a copy.
; CHECK-DAG: @both = private unnamed_addr constant [3 x i8] c"%s\00", align 1{{$}}
; CHECK-DAG: @both.arena = private global [3 x i8] c"%s\00", section ".addr_space.1", align 1
; CHECK-DAG: @ptr_to_both = global ptr @both.arena, section ".addr_space.1"
@both = private unnamed_addr constant [3 x i8] c"%s\00", align 1
@ptr_to_both = global ptr @both

; No copy of a constant whose address matters: 'p == named'.
; CHECK-DAG: @named = internal constant [2 x i8] c"a\00"{{$}}
@named = internal constant [2 x i8] c"a\00"

; The table of a computed goto has pointers to labels.
; CHECK-DAG: @labels = private unnamed_addr constant [2 x ptr] [ptr blockaddress(@jump, %a), ptr blockaddress(@jump, %b)]{{$}}
@labels = private unnamed_addr constant [2 x ptr] [ptr blockaddress(@jump, %a), ptr blockaddress(@jump, %b)]

; A kfunc fills it in, the program reads it: that has to be one variable.
; CHECK-DAG: @buf = internal global [16 x i8] zeroinitializer{{$}}
; CHECK-NOT: @buf.arena
@buf = internal global [16 x i8] zeroinitializer

; CHECK-DAG: @ext = external global i64{{$}}
@ext = external global i64

; The program declares the arena map.
; CHECK-NOT: section ".maps"

declare void @bpf_printk(ptr) section ".ksyms"
declare void @bpf_fill(ptr) section ".ksyms"
declare i8 @first(ptr)

define internal void @fn() {
  ret void
}

define i64 @use(i64 %n) {
; CHECK-LABEL: define i64 @use
; CHECK: call void @bpf_printk(ptr @fmt)
; CHECK: call void @bpf_printk(ptr @both)
; CHECK: call void @bpf_fill(ptr @buf)
; CHECK: call void @bpf_printk(ptr @named)
; CHECK: call i8 @first(ptr @named)
; CHECK: load i8, ptr @buf
  call void @bpf_printk(ptr @fmt)
  call void @bpf_printk(ptr @both)
  call void @bpf_fill(ptr @buf)
  call void @bpf_printk(ptr @named)
  %q = call i8 @first(ptr @named)
  %v = load i8, ptr @buf
  %e = getelementptr i64, ptr @table, i64 %n
  %c = load i64, ptr %e
  %d = load i64, ptr getelementptr (i8, ptr @direct, i64 8)
  %s = call i8 @first(ptr @str)
  %u = load ptr, ptr @uninit
  %f = load ptr, ptr @fns
  %g = load i32, ptr @config
  %r = add i64 %c, %d
  ret i64 %r
}

define i64 @jump(i64 %n) {
  %e = getelementptr ptr, ptr @labels, i64 %n
  %l = load ptr, ptr %e
  indirectbr ptr %l, [label %a, label %b]
a:
  ret i64 1
b:
  ret i64 2
}

; ELF-DAG: .jumptables       PROGBITS
; ELF-DAG: .addr_space.1     PROGBITS {{.*}} WA
; ELF-DAG: .rel.addr_space.1 REL
; ELF-DAG: .bss.addr_space.1 NOBITS   {{[0-9a-f]+}} {{[0-9a-f]+}} 002008 00  WA
; ELF-DAG: .rodata
; ELF-DAG: R_BPF_64_ABS64 {{.*}} zero

; ERR: error: 'p' is in arena and has a pointer to 'sec' that is not
