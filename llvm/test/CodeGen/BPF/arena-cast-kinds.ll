; RUN: opt -S -mtriple=bpfel -passes=bpf-arena-cast < %s | FileCheck %s
; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+arena < %s | FileCheck --check-prefix=ASM %s
; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+arena -enable-new-pm < %s | FileCheck --check-prefix=ASM %s

; A pointer is cast unless a pointer of the kernel is all that it can be. Where
; it comes from tells. Helper 158 is bpf_get_current_task_btf().

@in_arena = global ptr null, section ".bss.addr_space.1"
@in_rodata = constant i64 1
@table = internal constant [2 x ptr] [ptr @method, ptr @method2]
@"llvm.task_struct:0:8$0:1" = external global i64, !llvm.preserve.access.index !0 #0

declare ptr @bpf_task_from_pid(i32) section ".ksyms"
declare ptr addrspace(1) @bpf_alloc(i64, i64) section ".ksyms"
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

; The argument of a program is ctx.
define i64 @ctx(ptr %ctx) section "struct_ops/ctx" {
; CHECK-LABEL: @ctx(
; CHECK-NOT:   addrspacecast
; CHECK:       ret
; ASM-LABEL:   ctx:
; ASM-NOT:     addr_space_cast
; ASM:         exit
  %r = load i64, ptr %ctx
  ret i64 %r
}

; What is in ctx is not known. User space passes addresses in arena there.
define i64 @from_ctx(ptr %ctx) section "syscall" {
; CHECK-LABEL: @from_ctx(
; CHECK:       %p = load ptr, ptr %ctx
; CHECK-NEXT:  ptrtoint ptr %p to i64
; CHECK:       [[K:%.*]] = addrspacecast
; CHECK-NEXT:  load i64, ptr [[K]]
  %p = load ptr, ptr %ctx
  %r = load i64, ptr %p
  ret i64 %r
}

define i64 @helper() {
; CHECK-LABEL: @helper(
; CHECK-NOT:   addrspacecast
; CHECK:       ret
  %p = call ptr inttoptr (i64 158 to ptr)()
  %r = load i64, ptr %p
  ret i64 %r
}

; The type of this kfunc says that it returns memory in arena.
define i64 @kfunc_arena() {
; CHECK-LABEL: @kfunc_arena(
; CHECK:       %p.as1 = call ptr addrspace(1) @bpf_alloc
; CHECK:       %p = inttoptr
; CHECK-NEXT:  ptrtoint ptr %p to i64
; CHECK:       [[K:%.*]] = addrspacecast
; CHECK-NEXT:  load i64, ptr [[K]]
; ASM-LABEL:   kfunc_arena:
; ASM:         call bpf_alloc
; ASM-NEXT:    r1 = addr_space_cast(r0, 0, 1)
  %p.as1 = call ptr addrspace(1) @bpf_alloc(i64 8, i64 0)
  %p = addrspacecast ptr addrspace(1) %p.as1 to ptr
  %r = load i64, ptr %p
  ret i64 %r
}

; All other kfuncs return pointers of the kernel.
define i64 @kfunc_kernel() {
; CHECK-LABEL: @kfunc_kernel(
; CHECK-NOT:   addrspacecast
; CHECK:       ret
  %p = call ptr @bpf_task_from_pid(i32 1)
  %r = load i64, ptr %p
  ret i64 %r
}

; bpf_map_lookup_elem(). The map value is not cast. A pointer in there is what
; the program stored: an address in arena.
define i64 @map_value(ptr %map, ptr %key) {
; CHECK-LABEL: @map_value(
; CHECK:       %p = load ptr, ptr %v
; CHECK:       ptrtoint ptr %p to i64
; CHECK:       [[K:%.*]] = addrspacecast
; CHECK-NEXT:  load i64, ptr [[K]]
  %v = call ptr inttoptr (i64 1 to ptr)(ptr %map, ptr %key)
  %p = load ptr, ptr %v
  %r = load i64, ptr %p
  ret i64 %r
}

; A CO-RE relocation looks like a load from a variable of another file. It's
; not cast. The access at that offset is like other accesses: a type of
; the kernel can be a part of an object in arena.
define i64 @field(ptr %unknown) {
; CHECK-LABEL: @field(
; CHECK:       [[K:%.*]] = addrspacecast
; CHECK-NEXT:  %off = load i64, ptr @"llvm.task_struct:0:8$0:1"
; CHECK-NEXT:  [[F:%.*]] = getelementptr i8, ptr [[K]], i64 %off
; CHECK-NEXT:  load i64, ptr [[F]]
  %off = load i64, ptr @"llvm.task_struct:0:8$0:1"
  %f = getelementptr i8, ptr %unknown, i64 %off
  %r = load i64, ptr %f
  ret i64 %r
}

; A pointer of the kernel or NULL is a pointer of the kernel.
define i64 @kernel_or_null(i1 %c) {
; CHECK-LABEL: @kernel_or_null(
; CHECK-NOT:   addrspacecast
; CHECK:       ret
entry:
  br i1 %c, label %get, label %join
get:
  %t = call ptr inttoptr (i64 158 to ptr)()
  br label %join
join:
  %p = phi ptr [ null, %entry ], [ %t, %get ]
  %n = icmp eq ptr %p, null
  br i1 %n, label %out, label %deref
deref:
  %r = load i64, ptr %p
  ret i64 %r
out:
  ret i64 0
}

; An address in arena or NULL.
define i64 @arena_or_null(i1 %c) {
; CHECK-LABEL: @arena_or_null(
; CHECK:       %p = phi ptr
; CHECK-NEXT:  ptrtoint ptr %p to i64
; CHECK:       [[K:%.*]] = addrspacecast
; CHECK:       load i64, ptr [[K]]
entry:
  br i1 %c, label %get, label %join
get:
  %t.as1 = call ptr addrspace(1) @bpf_alloc(i64 8, i64 0)
  %t = addrspacecast ptr addrspace(1) %t.as1 to ptr
  br label %join
join:
  %p = phi ptr [ null, %entry ], [ %t, %get ]
  %n = icmp eq ptr %p, null
  br i1 %n, label %out, label %deref
deref:
  %r = load i64, ptr %p
  ret i64 %r
out:
  ret i64 0
}

define i64 @select_kernel(i1 %c) {
; CHECK-LABEL: @select_kernel(
; CHECK-NOT:   addrspacecast
; CHECK:       ret
  %t = call ptr inttoptr (i64 158 to ptr)()
  %p = select i1 %c, ptr %t, ptr @in_rodata
  %r = load i64, ptr %p
  ret i64 %r
}

define i64 @select_either(i1 %c) {
; CHECK-LABEL: @select_either(
; CHECK:       %p = select
; CHECK-NEXT:  ptrtoint ptr %p to i64
; CHECK:       [[K:%.*]] = addrspacecast
; CHECK-NEXT:  load i64, ptr [[K]]
  %t = call ptr inttoptr (i64 158 to ptr)()
  %m = load ptr, ptr @in_rodata
  %p = select i1 %c, ptr %t, ptr %m
  %r = load i64, ptr %p
  ret i64 %r
}

; The pointer is an integer for a while. Arithmetic makes it unknown.
define i64 @integer() {
; CHECK-LABEL: @integer(
; CHECK:       load i64, ptr %p
; CHECK:       ptrtoint ptr %q to i64
; CHECK:       [[K:%.*]] = addrspacecast
; CHECK-NEXT:  load i64, ptr [[K]]
  %t = call ptr inttoptr (i64 158 to ptr)()
  %i = ptrtoint ptr %t to i64
  %p = inttoptr i64 %i to ptr
  %a = load i64, ptr %p
  %j = and i64 %i, -2
  %q = inttoptr i64 %j to ptr
  %b = load i64, ptr %q
  %r = add i64 %a, %b
  ret i64 %r
}

; For the kernel a global variable that is not in arena is a map value.
define i64 @rodata() {
; CHECK-LABEL: @rodata(
; CHECK:       load i64, ptr @in_rodata
  %r = load i64, ptr @in_rodata
  ret i64 %r
}

; What is loaded from arena is an address in arena.
define i64 @from_arena() {
; CHECK-LABEL: @from_arena(
; CHECK:       %p = load ptr, ptr addrspacecast (ptr addrspace(1) inttoptr (i64 ptrtoint (ptr @in_arena to i64) to ptr addrspace(1)) to ptr)
; CHECK-NEXT:  ptrtoint ptr %p to i64
; CHECK:       [[K:%.*]] = addrspacecast
; CHECK-NEXT:  load i64, ptr [[K]]
  %p = load ptr, ptr @in_arena
  %r = load i64, ptr %p
  ret i64 %r
}

; All callers pass pointers of the kernel.
define internal i64 @gets_kernel(ptr %p) noinline {
; CHECK-LABEL: @gets_kernel(
; CHECK-NOT:   addrspacecast
; CHECK:       ret
  %r = load i64, ptr %p
  ret i64 %r
}

; All callers pass addresses in arena.
define internal i64 @gets_arena(ptr %p) noinline {
; CHECK-LABEL: @gets_arena(
; CHECK:       [[K:%.*]] = addrspacecast
; CHECK-NEXT:  load i64, ptr [[K]]
  %r = load i64, ptr %p
  ret i64 %r
}

; The caller that passes a pointer of the kernel gets a copy.
define internal i64 @gets_both(ptr %p) noinline {
; CHECK-LABEL: @gets_both(
; CHECK:       [[K:%.*]] = addrspacecast
; CHECK-NEXT:  load i64, ptr [[K]]
  %r = load i64, ptr %p
  ret i64 %r
}

; Not all callers are seen.
define i64 @gets_more(ptr %p) noinline {
; CHECK-LABEL: @gets_more(
; CHECK:       [[K:%.*]] = addrspacecast
; CHECK-NEXT:  load i64, ptr [[K]]
  %r = load i64, ptr %p
  ret i64 %r
}

; It returns a pointer of the kernel.
define internal ptr @get_task() noinline {
  %t = call ptr inttoptr (i64 158 to ptr)()
  ret ptr %t
}

define i64 @calls() {
; CHECK-LABEL: @calls(
; CHECK-NOT:   addrspacecast
; CHECK:       call i64 @gets_both.kern(ptr %t)
; CHECK-NEXT:  call i64 @gets_both(ptr %m)
; CHECK-NOT:   addrspacecast
; CHECK:       ret
  %t = call ptr @get_task()
  %m.as1 = call ptr addrspace(1) @bpf_alloc(i64 8, i64 0)
  %m = addrspacecast ptr addrspace(1) %m.as1 to ptr
  %a = call i64 @gets_kernel(ptr %t)
  %b = call i64 @gets_arena(ptr %m)
  %c = call i64 @gets_both(ptr %t)
  %d = call i64 @gets_both(ptr %m)
  %e = call i64 @gets_more(ptr %t)
  %f = load i64, ptr %t
  ret i64 %f
}

; Its address is taken. The indirect calls of its type pass pointers of
; the kernel.
define internal i64 @method(ptr %p) {
; CHECK-LABEL: @method(
; CHECK-NOT:   addrspacecast
; CHECK:       ret
  %r = load i64, ptr %p
  ret i64 %r
}

; The indirect call of this type passes an address in arena.
define internal i32 @method2(ptr %p) {
; CHECK-LABEL: @method2(
; CHECK:       [[K:%.*]] = addrspacecast
; CHECK-NEXT:  load i32, ptr [[K]]
  %r = load i32, ptr %p
  ret i32 %r
}

define i64 @indirect() {
  %t = call ptr inttoptr (i64 158 to ptr)()
  %f = load ptr, ptr @table
  %r = call i64 %f(ptr %t)
  %m.as1 = call ptr addrspace(1) @bpf_alloc(i64 8, i64 0)
  %m = addrspacecast ptr addrspace(1) %m.as1 to ptr
  %g = load ptr, ptr getelementptr (i8, ptr @table, i64 8)
  %s = call i32 %g(ptr %m)
  ret i64 %r
}

; The kernel calls it with pointers of its own.
define internal i64 @callback(ptr %map, ptr %key, ptr %value) {
; CHECK-LABEL: @callback(
; CHECK-NOT:   addrspacecast
; CHECK:       ret
  %r = load i64, ptr %key
  ret i64 %r
}

; bpf_timer_set_callback()
define i64 @set_callback(ptr %timer) {
  %r = call i64 inttoptr (i64 170 to ptr)(ptr %timer, ptr @callback)
  ret i64 %r
}

; The one call that is seen passes a pointer of the kernel. The other one
; appears when insns are selected.
define internal ptr @memset(ptr %dst, i32 %c, i64 %n) noinline {
; CHECK-LABEL: @memset(
; CHECK:       [[K:%.*]] = addrspacecast
; CHECK:       store i8 %b, ptr [[K]]
  %b = trunc i32 %c to i8
  store i8 %b, ptr %dst
  ret ptr %dst
}

define void @calls_memset(i64 %n) {
; ASM-LABEL:   calls_memset:
; ASM:         call memset
; ASM:         call memset
  %t = call ptr inttoptr (i64 158 to ptr)()
  %r = call ptr @memset(ptr %t, i32 0, i64 8)
  %m.as1 = call ptr addrspace(1) @bpf_alloc(i64 %n, i64 0)
  %m = addrspacecast ptr addrspace(1) %m.as1 to ptr
  call void @llvm.memset.p0.i64(ptr %m, i8 0, i64 %n, i1 false)
  ret void
}

; freeze keeps the kind.
define i64 @frozen() {
; CHECK-LABEL: @frozen(
; CHECK-NOT:   addrspacecast
; CHECK:       ret
  %p = call ptr @bpf_task_from_pid(i32 1)
  %f = freeze ptr %p
  %r = load i64, ptr %f
  ret i64 %r
}

; An access through NULL is an access to arena.
define i64 @null() {
; CHECK-LABEL: @null(
; CHECK:       load i64, ptr addrspacecast (ptr addrspace(1) null to ptr)
; ASM-LABEL:   null:
; ASM:         r1 = 0
; ASM-NEXT:    r1 = addr_space_cast(r1, 0, 1)
; ASM-NEXT:    r0 = *(u64 *)(r1 + 0)
  %r = load i64, ptr null
  ret i64 %r
}

define i64 @null_index(i64 %n) {
; CHECK-LABEL: @null_index(
; CHECK:       [[P:%.*]] = getelementptr i8, ptr addrspacecast (ptr addrspace(1) null to ptr), i64 %n
; CHECK-NEXT:  load i64, ptr [[P]]
  %p = getelementptr i8, ptr null, i64 %n
  %r = load i64, ptr %p
  ret i64 %r
}

attributes #0 = { "btf_ama" }
!0 = !{}
