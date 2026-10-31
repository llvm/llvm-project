; RUN: opt -mtriple=bpfel -passes=bpf-arena-cast -S < %s | FileCheck %s
; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+arena-cast < %s | FileCheck --check-prefix=ASM %s

; How a pointer that can be an address in arena is cast. arena-cast-kinds.ll:
; which pointers that are.

@g = global [4 x i64] zeroinitializer, section ".bss.addr_space.1"
@"llvm.s:0:8$0:1" = external global i64, !llvm.preserve.access.index !0 #0

define void @take(ptr %p) noinline {
  ret void
}

; One cast for two accesses. The offsets are added on top of it.
define i64 @arg(ptr %p) {
; CHECK-LABEL: @arg
; CHECK:      [[I:%.*]] = ptrtoint ptr %p to i64
; CHECK-NEXT: [[U:%.*]] = inttoptr i64 [[I]] to ptr addrspace(1)
; CHECK-NEXT: [[K:%.*]] = addrspacecast ptr addrspace(1) [[U]] to ptr
; CHECK-NEXT: [[A:%.*]] = load i64, ptr [[K]]
; CHECK-NEXT: [[Q:%.*]] = getelementptr i8, ptr [[K]], i64 8
; CHECK-NEXT: store i64 [[A]], ptr [[Q]]
; CHECK-NOT:  addrspacecast
; ASM-LABEL: arg:
; ASM:         r1 = addr_space_cast(r1, 0, 1)
; ASM-NEXT:    r0 = *(u64 *)(r1 + 0)
; ASM-NEXT:    *(u64 *)(r1 + 8) = r0
; ASM-NEXT:    exit
  %a = load i64, ptr %p
  %q = getelementptr i8, ptr %p, i64 8
  store i64 %a, ptr %q
  ret i64 %a
}

; What is passed, stored or compared is the pointer as it was.
define i1 @value(ptr %p, ptr %out) {
; CHECK-LABEL: @value
; CHECK:      call void @take(ptr %p)
; CHECK:      store ptr %p, ptr
; CHECK:      icmp eq ptr %p, %out
  %a = load i64, ptr %p
  call void @take(ptr %p)
  store ptr %p, ptr %out
  %c = icmp eq ptr %p, %out
  ret i1 %c
}

; The pointer that is loaded is cast where it's defined.
define i64 @loaded(ptr %p) {
; CHECK-LABEL: @loaded
; CHECK:      [[Q:%.*]] = load ptr, ptr
; CHECK-NEXT: [[I:%.*]] = ptrtoint ptr [[Q]] to i64
; CHECK-NEXT: inttoptr i64 [[I]] to ptr addrspace(1)
; ASM-LABEL: loaded:
; ASM:         r1 = addr_space_cast(r1, 0, 1)
; ASM-NEXT:    r1 = *(u64 *)(r1 + 0)
; ASM-NEXT:    r1 = addr_space_cast(r1, 0, 1)
; ASM-NEXT:    r0 = *(u64 *)(r1 + 0)
  %q = load ptr, ptr %p
  %r = load i64, ptr %q
  ret i64 %r
}

; Address arithmetic that is only for the access is done on top of the cast.
define i64 @index(ptr %p, i64 %i) {
; CHECK-LABEL: @index
; CHECK:      [[K:%.*]] = addrspacecast
; CHECK-NEXT: [[Q:%.*]] = getelementptr i64, ptr [[K]], i64 %i
; CHECK-NEXT: load i64, ptr [[Q]]
; CHECK-NOT:  getelementptr
  %q = getelementptr i64, ptr %p, i64 %i
  %r = load i64, ptr %q
  ret i64 %r
}

; Its result is needed as a value too: it's computed once and cast.
define i64 @index_value(ptr %p, i64 %i) {
; CHECK-LABEL: @index_value
; CHECK:      %q = getelementptr i64, ptr %p, i64 %i
; CHECK-NEXT: [[I:%.*]] = ptrtoint ptr %q to i64
; CHECK:      [[K:%.*]] = addrspacecast
; CHECK-NEXT: load i64, ptr [[K]]
; CHECK-NEXT: call void @take(ptr %q)
  %q = getelementptr i64, ptr %p, i64 %i
  %r = load i64, ptr %q
  call void @take(ptr %q)
  ret i64 %r
}

; A global variable in arena. The offset is in the insn that accesses memory.
define i64 @global() {
; CHECK-LABEL: @global
; CHECK: load i64, ptr getelementptr (i8, ptr addrspacecast (ptr addrspace(1) inttoptr (i64 ptrtoint (ptr @g to i64) to ptr addrspace(1)) to ptr), i64 16)
; ASM-LABEL: global:
; ASM:         r1 = g ll
; ASM-NEXT:    r1 = addr_space_cast(r1, 0, 1)
; ASM-NEXT:    r0 = *(u64 *)(r1 + 16)
  %r = load i64, ptr getelementptr (i8, ptr @g, i64 16)
  ret i64 %r
}

define void @atomics(ptr %p) {
; CHECK-LABEL: @atomics
; CHECK:      [[K:%.*]] = addrspacecast
; CHECK-NEXT: atomicrmw add ptr [[K]]
; CHECK-NEXT: cmpxchg ptr [[K]]
  %a = atomicrmw add ptr %p, i64 1 seq_cst
  %b = cmpxchg ptr %p, i64 0, i64 1 seq_cst seq_cst
  ret void
}

define void @copy(ptr %d, ptr %s) {
; CHECK-LABEL: @copy
; CHECK-DAG:  [[D:%.*]] = ptrtoint ptr %d to i64
; CHECK-DAG:  [[S:%.*]] = ptrtoint ptr %s to i64
; CHECK:      call void @llvm.memcpy.p0.p0.i64(ptr [[KD:%[0-9]+]], ptr [[KS:%[0-9]+]], i64 16, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 16, i1 false)
  ret void
}

; The stack of the program is not arena.
define i64 @stack() {
; CHECK-LABEL: @stack
; CHECK-NOT:  addrspacecast
; CHECK:      ret
  %a = alloca i64
  store volatile i64 1, ptr %a
  %r = load volatile i64, ptr %a
  ret i64 %r
}

; It's cast in the source.
define i64 @arena(ptr addrspace(1) %p) {
; CHECK-LABEL: @arena
; CHECK:      %k = addrspacecast ptr addrspace(1) %p to ptr
; CHECK-NEXT: load i64, ptr %k
; CHECK-NOT:  addrspacecast
  %k = addrspacecast ptr addrspace(1) %p to ptr
  %r = load i64, ptr %k
  ret i64 %r
}

; Not memory: a CO-RE relocation looks like a load until it's lowered.
define i64 @core() {
; CHECK-LABEL: @core
; CHECK: load i64, ptr @"llvm.s:0:8$0:1"
  %r = load i64, ptr @"llvm.s:0:8$0:1"
  ret i64 %r
}

; There is no place after the definition. The cast is at the access.
define i64 @invoke_result(i1 %c) uwtable personality ptr null {
; CHECK-LABEL: @invoke_result
; CHECK:      join:
; CHECK-NEXT:   [[I:%.*]] = ptrtoint ptr %p to i64
; CHECK:        addrspacecast
entry:
  %p = invoke ptr @make() to label %join unwind label %pad
pad:
  %lp = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %lp
join:
  %r = load i64, ptr %p
  ret i64 %r
}

define ptr @make() noinline {
  ret ptr null
}

attributes #0 = { "btf_ama" }
!0 = !{}
