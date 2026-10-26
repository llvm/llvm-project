; RUN: opt -mtriple=bpfel -passes=bpf-arena-data -S < %s | FileCheck %s
; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+arena-data -filetype=obj -o %t < %s
; RUN: llvm-readelf -S -r %t | FileCheck --check-prefix=ELF %s
; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+arena-data < %s | FileCheck --check-prefix=ASM %s
; RUN: not opt -mtriple=bpfel -passes=bpf-arena-data -disable-output %S/Inputs/arena-data-err.ll 2>&1 \
; RUN:   | FileCheck --check-prefix=ERR %s
; RUN: opt -mtriple=bpfel -passes=bpf-arena-data -S %S/Inputs/arena-data-map.ll \
; RUN:   | FileCheck --check-prefix=MAP %s

; Global data is in arena.

; Variables, constants and strings that are not zero are in one section.
; CHECK-DAG: @var = global i64 5, section ".addr_space.1"
; CHECK-DAG: @const = internal global i64 7, section ".addr_space.1"
; CHECK-DAG: @str = private global [6 x i8] c"hello\00", section ".addr_space.1", align 1
@var = global i64 5
@const = internal constant i64 7
@str = private unnamed_addr constant [6 x i8] c"hello\00", align 1

; Those that are zero take no space in the file.
; CHECK-DAG: @zero = global [1024 x i64] zeroinitializer, section ".bss.addr_space.1"
; CHECK-DAG: @undef = internal global ptr undef, section ".bss.addr_space.1"
@zero = global [1024 x i64] zeroinitializer
@undef = internal global ptr undef

; The loader resolves pointers.
; CHECK-DAG: @ptr = global ptr getelementptr {{.*}}(i8, ptr @zero, i64 8), section ".addr_space.1"
@ptr = global ptr getelementptr (i8, ptr @zero, i64 8)

; What has a section stays there.
; CHECK-DAG: @_license = global [4 x i8] c"GPL\00", section "license"
@_license = global [4 x i8] c"GPL\00", section "license"

; The kernel looks for pointers to functions in read-only data.
; CHECK-DAG: @fns = internal constant [1 x ptr] {{\[}}ptr @use]{{$}}
@fns = internal constant [1 x ptr] [ptr @use]

; A kfunc takes the string from a read-only map.
; CHECK-DAG: @fmt = private unnamed_addr constant [3 x i8] c"%d\00", align 1{{$}}
@fmt = private unnamed_addr constant [3 x i8] c"%d\00", align 1

; It does, the program reads a copy.
; CHECK-DAG: @both = private unnamed_addr constant [3 x i8] c"%s\00", align 1{{$}}
; CHECK-DAG: @both.arena = private global [3 x i8] c"%s\00", section ".addr_space.1", align 1
; CHECK-DAG: @ptr_to_both = global ptr @both.arena, section ".addr_space.1"
@both = private unnamed_addr constant [3 x i8] c"%s\00", align 1
@ptr_to_both = global ptr @both

; A kfunc fills it in, the program reads it: that has to be one variable.
; CHECK-DAG: @buf = internal global [16 x i8] zeroinitializer{{$}}
; CHECK-NOT: @buf.arena
@buf = internal global [16 x i8] zeroinitializer

; CHECK-DAG: @ext = external global i64{{$}}
@ext = external global i64

; There is no arena map, add one.
; CHECK-DAG: @arena = global { ptr, ptr } zeroinitializer, section ".maps", align 8, !dbg [[GVE:![0-9]+]]
; CHECK-DAG: @llvm.compiler.used = appending global [1 x ptr] {{\[}}ptr @arena]

declare void @bpf_printk(ptr)
declare void @bpf_fill(ptr)

define i64 @use() {
; CHECK-LABEL: define i64 @use
; CHECK: call void @bpf_printk(ptr @fmt)
; CHECK: call void @bpf_printk(ptr @both)
; CHECK: load i8, ptr @both.arena
; CHECK: call void @bpf_fill(ptr @buf)
; CHECK: load i8, ptr @buf
  call void @bpf_printk(ptr @fmt)
  call void @bpf_printk(ptr @both)
  %b = load i8, ptr @both
  call void @bpf_fill(ptr @buf)
  %v = load i8, ptr @buf
  %c = load i64, ptr @const
  %s = load i8, ptr @str
  %u = load ptr, ptr @undef
  %f = load ptr, ptr @fns
  ret i64 %c
}

; int (*type)[33], int (*map_flags)[1024]
; CHECK: [[GVE]] = !DIGlobalVariableExpression(var: [[VAR:![0-9]+]]
; CHECK: [[VAR]] = distinct !DIGlobalVariable(name: "arena", {{.*}}type: [[TY:![0-9]+]], isLocal: false, isDefinition: true)
; CHECK: [[TY]] = !DICompositeType(tag: DW_TAG_structure_type, {{.*}}size: 128, {{.*}}elements: [[ELTS:![0-9]+]])
; CHECK: [[ELTS]] = !{[[TYPE:![0-9]+]], [[FLAGS:![0-9]+]]}
; CHECK: [[TYPE]] = !DIDerivedType(tag: DW_TAG_member, name: "type", {{.*}}baseType: [[P1:![0-9]+]], size: 64
; CHECK: !DISubrange(count: 33
; CHECK: [[FLAGS]] = !DIDerivedType(tag: DW_TAG_member, name: "map_flags", {{.*}}offset: 64)
; CHECK: !DISubrange(count: 1024

; ELF-DAG: .addr_space.1     PROGBITS {{.*}} WA
; ELF-DAG: .rel.addr_space.1 REL
; ELF-DAG: .bss.addr_space.1 NOBITS   {{[0-9a-f]+}} {{[0-9a-f]+}} 002008 00  WA
; ELF-DAG: .maps             PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} 000010 00  WA
; ELF-DAG: R_BPF_64_ABS64 {{.*}} zero

; The map is in BTF.
; ASM: .ascii "arena"
; ASM: .ascii ".maps"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

; ERR: error: 'p' is in arena and has a pointer to 'sec' that is not

; The program has declared the map, with the size.
; MAP-NOT: @arena =
; MAP: @my_arena = {{.*}}section ".maps"
; MAP-NOT: @arena =
