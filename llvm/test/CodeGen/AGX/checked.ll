; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+agx-checked -stop-after=agx-checked-accesses < %s | FileCheck %s --check-prefix=IR
; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+agx-checked -verify-machineinstrs < %s | FileCheck %s --check-prefix=REC
; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+agx-checked,+agx-strict -verify-machineinstrs -filetype=obj < %s -o /dev/null
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s --check-prefix=OFF

; Checked mode (AGXUsage.rst): every device, threadgroup and variable stack
; access is checked before it runs; the bounds of argument-based pointers and
; the error record are hidden arguments.

@g = addrspace(1) global [16 x i32] zeroinitializer
@tg = internal addrspace(3) global [64 x i32] undef

declare i32 @llvm.agx.thread.position.in.threadgroup.x()

; One check site per access, in order.
; IR: @__agx_checked_sites = addrspace(1) constant {{.*}}c"1\09load 4\09k\0A2\09load 4\09k\0A3\09store 4\09k\0A4\09store 4\09k\0A5\09store 4\09k\0A6\09atomic 4\09k\0A\00"

; The hidden arguments follow the source ones: record, threadgroup bytes, then
; the bounds of each slot (the pointer field at offset 0 of %params, then %out).
; IR-LABEL: define agx_kernel void @k(ptr addrspace(1) %out, ptr addrspace(4) byref([16 x i8]) %params, i32 %n,
; IR-SAME:      ptr addrspace(1) "agx-checked"="record" [[RECORD:%[0-9]+]], i32 "agx-checked"="threadgroup-bytes" [[TGB:%[0-9]+]],
; IR-SAME:      i64 "agx-checked"="lo0" [[LO0:%[0-9]+]], i64 "agx-checked"="hi0" [[HI0:%[0-9]+]], i64 "agx-checked"="lo1" [[LO1:%[0-9]+]], i64 "agx-checked"="hi1" [[HI1:%[0-9]+]])
; A recorded error makes the kernel return at once.
; IR:        %claimed = load i32, ptr addrspace(1) [[RECORD]]
; IR:        br i1 %{{.*}}, label %checked.exit,
; Slot 0, the pointer loaded from %params: alignment and bounds (which also
; catch null: a pointer outside every allocation has empty bounds).
; IR:        [[A:%[0-9]+]] = ptrtoint ptr addrspace(1) %pa to i64
; IR:        and i32 %{{.*}}, 3
; IR:        icmp ult i64 [[A]], [[LO0]]
; IR:        icmp ugt i64 %{{.*}}, [[HI0]]
; IR:        br i1 %{{.*}}, label %checked.report,
; IR:        %x = load i32
; A module global: one compare of the offset in it.
; IR:        sub i64 %{{.*}}, ptrtoint (ptr addrspace(1) @g to i64)
; IR-NEXT:   icmp ugt i64 %{{.*}}, 60
; A static threadgroup array likewise, in 32 bits.
; IR:        sub i32 %{{.*}}, ptrtoint (ptr addrspace(3) @tg to i32)
; IR-NEXT:   icmp ugt i32 %{{.*}}, 252
; IR:        store i32 %x, ptr addrspace(3) %ptg
; The stack object with a variable index against its 32 bytes, once: the load
; after the store reads the bytes it checked.
; IR:        icmp sgt i64 %{{.*}}, 32
; IR:        store i32 %y, ptr addrspace(5) %ps
; IR-NEXT:   %z = load i32, ptr addrspace(5) %ps
; Slot 1 (%out).
; IR:        icmp ugt i64 %{{.*}}, [[HI1]]
; IR:        store i32 %z
; IR:        atomicrmw add ptr addrspace(1) %out
; The first failing thread claims and fills the record.
; IR:      checked.report:
; The kinds are classes until the report: 1 device, 4 threadgroup, 5 stack;
; bit 8 a store, bit 9 an atomic.
; IR-NEXT:   %class = phi i32 [ 1, {{.*}} ], [ 1, {{.*}} ], [ 260, {{.*}} ], [ 261, {{.*}} ], [ 257, {{.*}} ], [ 513, {{.*}} ]
; IR:        cmpxchg ptr addrspace(1) [[RECORD]], i32 0, i32 1 monotonic monotonic
; IR:      checked.fill:
; IR:        call i32 @llvm.agx.thread.position.in.threadgroup.x()
; IR:        call i32 @llvm.agx.threadgroup.position.in.grid.x()
; IR:      checked.exit:
; IR-NEXT:   ret void

; The record: 3 source arguments, then extension 2: version 1, 2 slots, the
; record at half 14, threadgroup bytes at 18; slot 0 is argument 1 at offset
; 0 (bounds at halves 20, 24), slot 1 argument 0 itself (28, 32).
; REC:      .section .agx.kernels
; REC:      .long 80
; REC-NEXT: .long 3
; REC-NEXT: .long 0
; REC-NEXT: .long 8
; REC-NEXT: .long 0
; REC-NEXT: .long 0
; REC-NEXT: .long 16
; REC-NEXT: .long 4
; REC-NEXT: .long 0
; REC-NEXT: .long 4
; REC-NEXT: .long 12
; REC-NEXT: .long 2
; REC-NEXT: .long 1
; REC-NEXT: .long 2
; REC-NEXT: .long 14
; REC-NEXT: .long 18
; REC-NEXT: .long 1
; REC-NEXT: .long 0
; REC-NEXT: .long 20
; REC-NEXT: .long 24
; REC-NEXT: .long 0
; REC-NEXT: .long 4294967295
; REC-NEXT: .long 28
; REC-NEXT: .long 32
; REC-NEXT: .section

; Without the feature nothing changes.
; OFF-NOT: __agx_checked_sites
; OFF-NOT: cmpxchg

define agx_kernel void @k(ptr addrspace(1) %out, ptr addrspace(4) byref([16 x i8]) %params, i32 %n) {
entry:
  %stack = alloca [8 x i32], align 4, addrspace(5)
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %pin = load ptr addrspace(1), ptr addrspace(4) %params, align 8
  %pa = getelementptr i32, ptr addrspace(1) %pin, i32 %t
  %x = load i32, ptr addrspace(1) %pa, align 4
  %pg = getelementptr [16 x i32], ptr addrspace(1) @g, i32 0, i32 %t
  %y = load i32, ptr addrspace(1) %pg, align 4
  %ptg = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %t
  store i32 %x, ptr addrspace(3) %ptg, align 4
  %idx = and i32 %n, 15
  %ps = getelementptr [8 x i32], ptr addrspace(5) %stack, i32 0, i32 %idx
  store i32 %y, ptr addrspace(5) %ps, align 4
  %z = load i32, ptr addrspace(5) %ps, align 4
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %t
  store i32 %z, ptr addrspace(1) %po, align 4
  %old = atomicrmw add ptr addrspace(1) %out, i32 1 monotonic
  ret void
}
