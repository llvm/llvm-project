; RUN: opt -S -mtriple=bpfel -passes=bpf-arena-data < %s | FileCheck %s

; A variable that gets a pointer of the kernel stays. What is a pointer of
; the kernel depends on which variables move.

; CHECK-DAG: @saved = internal global ptr null{{$}}
; CHECK-DAG: @filled = internal global ptr null{{$}}
@saved = internal global ptr null
@filled = internal global ptr null

; The program keeps what it loads as a number, uses it as an index, or uses
; another field of the variable.
; CHECK-DAG: @number = internal global i64 0{{$}}
; CHECK-DAG: @fields = internal global [2 x i64] zeroinitializer{{$}}
@number = internal global i64 0
@fields = internal global [2 x i64] zeroinitializer
@copy = internal global i64 0
@hist = internal global [16 x i64] zeroinitializer

; Outside of arena the pointer comes back as a number too. The program uses it
; for an access, or can do so where it is not seen: the variable moves, and
; the verifier can reject the store.
; CHECK-DAG: @used = internal global ptr null, section ".bss.addr_space.1"
; CHECK-DAG: @used_by_callee = internal global ptr null, section ".bss.addr_space.1"
; CHECK-DAG: @known = internal global ptr null, section ".bss.addr_space.1"
; CHECK-DAG: @pknown = internal global ptr null, section ".bss.addr_space.1"
@used = internal global ptr null
@used_by_callee = internal global ptr null
@known = internal global ptr null
@pknown = internal global ptr null

; The addresses of a variable of the kernel, of a map, of a variable that
; a kfunc gets and of a constant that the loader can set.
; CHECK-DAG: @k1 = internal global ptr null{{$}}
; CHECK-DAG: @k2 = internal global ptr null{{$}}
; CHECK-DAG: @k3 = internal global ptr null{{$}}
; CHECK-DAG: @k4 = internal global ptr null{{$}}
@ksym = external global i64, section ".ksyms"
@map = global i64 0, section ".maps"
@buf = internal global [16 x i8] zeroinitializer
@config = constant [2 x i32] [i32 1, i32 2]
@k1 = internal global ptr null
@k2 = internal global ptr null
@k3 = internal global ptr null
@k4 = internal global ptr null

; Addresses in arena are fine in arena: a variable that is there already, one of
; another file, one that moves.
; CHECK-DAG: @a1 = internal global ptr null, section ".bss.addr_space.1"
; CHECK-DAG: @a2 = internal global ptr null, section ".bss.addr_space.1"
; CHECK-DAG: @a3 = internal global ptr null, section ".bss.addr_space.1"
@marked = addrspace(1) global i64 0
@ext = external global i64
@var = internal global i64 0
@a1 = internal global ptr null
@a2 = internal global ptr null
@a3 = internal global ptr null

declare ptr @bpf_task_get() section ".ksyms"
declare void @bpf_fill(ptr) section ".ksyms"

define internal void @fill(ptr %p, ptr %t) noinline {
  store ptr %t, ptr %p
  ret void
}

define internal i64 @read(ptr %p) noinline {
  %t = load ptr, ptr %p
  %v = load i64, ptr %t
  ret i64 %v
}

define void @prog(ptr %ctx) section "syscall" {
  %t = call ptr @bpf_task_get()
  store ptr %t, ptr @saved
  %same = load ptr, ptr @saved
  %cmp = icmp eq ptr %same, %t
  call void @fill(ptr @filled, ptr %t)
  %ti = ptrtoint ptr %t to i64
  store i64 %ti, ptr @number
  %num = load i64, ptr @number
  store i64 %num, ptr @copy
  %idx = and i64 %num, 15
  %slot = getelementptr i64, ptr @hist, i64 %idx
  store i64 1, ptr %slot
  store i64 %ti, ptr getelementptr (i64, ptr @fields, i64 1)
  %cnt = atomicrmw add ptr @fields, i64 1 seq_cst
  %cp = inttoptr i64 %cnt to ptr
  %cv = load i64, ptr %cp
  store ptr %t, ptr @used
  %u = load ptr, ptr @used
  %v = load i64, ptr %u
  store ptr %t, ptr @used_by_callee
  %w = call i64 @read(ptr @used_by_callee)
  store ptr %t, ptr @known
  store ptr @known, ptr @pknown
  call void @bpf_fill(ptr @buf)
  store ptr @ksym, ptr @k1
  store ptr @map, ptr @k2
  store ptr @buf, ptr @k3
  store ptr @config, ptr @k4
  store ptr addrspacecast (ptr addrspace(1) @marked to ptr), ptr @a1
  store ptr @ext, ptr @a2
  store ptr @var, ptr @a3
  ret void
}
