; RUN: llc -mtriple=bpfel -mcpu=v1 -bpf-stack-spilling -verify-machineinstrs < %s | FileCheck %s -D#FRAME=4096
; RUN: llc -mtriple=bpfel -mcpu=v3 -bpf-stack-spilling -verify-machineinstrs < %s | FileCheck %s -D#FRAME=4096
; RUN: llc -mtriple=bpfeb -mcpu=v3 -bpf-stack-spilling -verify-machineinstrs < %s | FileCheck %s -D#FRAME=4096
; RUN: llc -mtriple=bpfel -mcpu=v4 -bpf-stack-spilling -verify-machineinstrs < %s | FileCheck %s -D#FRAME=4096
; RUN: llc -O0 -mtriple=bpfel -mcpu=v1 -bpf-stack-spilling -verify-machineinstrs < %s | FileCheck %s -D#FRAME=4096
; RUN: llc -O0 -mtriple=bpfel -mcpu=v3 -bpf-stack-spilling -verify-machineinstrs < %s | FileCheck %s -D#FRAME=4096
; RUN: llc -mtriple=bpfel -mcpu=v3 -bpf-stack-spilling -bpf-stack-size=512 -verify-machineinstrs < %s | FileCheck %s -D#FRAME=512
; RUN: llc -mtriple=bpfel -mcpu=v3 -bpf-stack-spilling -bpf-stack-size=32768 -verify-machineinstrs < %s | FileCheck %s -D#FRAME=32768
; RUN: llc -O0 -mtriple=bpfel -mcpu=v3 -bpf-stack-spilling -filetype=obj < %s -o %t
; RUN: llvm-objdump -d %t | FileCheck %s --check-prefix=OBJ

; This is a stack ABI option, independent of the Linux BPF instruction set
; selected with -mcpu. Calls must not use r11 or adjust r10 in this mode.
; OBJ-NOT: r11
; OBJ-NOT: store_stack_arg
; OBJ-NOT: load_stack_arg
; OBJ: call
; OBJ-NOT: r11

declare i64 @ext(i64, i64, i64, i64, i64, i64, i64)
declare void @use(ptr)

define i64 @caller(i64 %x, i64 %y) {
; CHECK-LABEL: caller:
; CHECK-NOT: r11
; CHECK-DAG: *(u64 *)(r10 + 0) = r1
; CHECK-DAG: *(u64 *)(r10 + 8) = r2
; CHECK-DAG: {{[rw]}}5 = 5
; CHECK: call ext
; CHECK-NOT: r10 +=
; CHECK: exit
  %r = call i64 @ext(i64 1, i64 2, i64 3, i64 4, i64 5, i64 %x, i64 %y)
  ret i64 %r
}

define i64 @callee(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %x, i64 %y) {
; CHECK-LABEL: callee:
; CHECK-NOT: r11
; CHECK-DAG: r[[X:[0-9]+]] = *(u64 *)(r10 - [[#FRAME]])
; CHECK-DAG: r[[Y:[0-9]+]] = *(u64 *)(r10 - [[#FRAME-8]])
; CHECK: r[[X]] += r[[Y]]
; CHECK: exit
  %r = add i64 %x, %y
  ret i64 %r
}

; Incoming arguments reserve space before locals, while outgoing arguments
; belong to the next frame and must not overlap either of them.
define i64 @forward(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %x, i64 %y) {
; CHECK-LABEL: forward:
; CHECK-NOT: r11
; CHECK-DAG: r{{[0-9]+}} = *(u64 *)(r10 - [[#FRAME]])
; CHECK-DAG: r{{[0-9]+}} = *(u64 *)(r10 - [[#FRAME-8]])
; CHECK-DAG: *(u64 *)(r10 - [[#FRAME-16]]) = {{r[0-9]+|42}}
; CHECK-DAG: *(u64 *)(r10 + 0) = r{{[0-9]+}}
; CHECK-DAG: *(u64 *)(r10 + 8) = r{{[0-9]+}}
; CHECK: call ext
; CHECK: r{{[0-9]+}} = *(u64 *)(r10 - [[#FRAME-16]])
; CHECK: exit
  %local = alloca i64, align 8
  store volatile i64 42, ptr %local
  %r = call i64 @ext(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %x, i64 %y)
  %v = load volatile i64, ptr %local
  %sum = add i64 %r, %v
  ret i64 %sum
}

; A frame index used as an address must use the same layout as loads/stores.
define void @local_address() {
; CHECK-LABEL: local_address:
; CHECK: r1 = r10
; CHECK-NEXT: r1 += -[[#FRAME]]
; CHECK: call use
; CHECK: exit
  %local = alloca i64, align 8
  call void @use(ptr %local)
  ret void
}

; Repeated calls reuse the next frame's argument area. Neither call's stores
; may disappear, even when the callee cannot otherwise access the caller frame.
define void @two_calls(i64 %x, i64 %y) {
; CHECK-LABEL: two_calls:
; CHECK-DAG: *(u64 *)(r10 + 0) = r{{[0-9]+}}
; CHECK-DAG: *(u64 *)(r10 + 8) = r{{[0-9]+}}
; CHECK: call ext
; CHECK-DAG: *(u64 *)(r10 + 0) = r{{[0-9]+}}
; CHECK-DAG: *(u64 *)(r10 + 8) = r{{[0-9]+}}
; CHECK: call ext
; CHECK: exit
  %r1 = call i64 @ext(i64 1, i64 2, i64 3, i64 4, i64 5, i64 %x, i64 %y)
  %r2 = call i64 @ext(i64 1, i64 2, i64 3, i64 4, i64 5, i64 %y, i64 %x)
  ret void
}

; An aggregate split between r5 and the stack keeps that register assignment.
define i64 @split_aggregate(i64 %a, i64 %b, i64 %c, i64 %d, [2 x i64] %pair) {
; CHECK-LABEL: split_aggregate:
; CHECK-DAG: r{{[0-9]+}} = *(u64 *)(r10 - [[#FRAME]])
; CHECK-DAG: {{r[0-9]+ = r5|r[0-9]+ \+= r5|r5 \+= r[0-9]+}}
; CHECK: exit
  %lo = extractvalue [2 x i64] %pair, 0
  %hi = extractvalue [2 x i64] %pair, 1
  %r = add i64 %lo, %hi
  ret i64 %r
}

; Register allocator spills must use the current frame, even when outgoing
; arguments occupy the next frame at nonnegative offsets.
define i64 @spills(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e) {
; CHECK-LABEL: spills:
; CHECK: *(u64 *)(r10 - [[#FRAME]]) = r{{[0-9]+}}
; CHECK: call ext
; CHECK: r{{[0-9]+}} = *(u64 *)(r10 - [[#FRAME]])
; CHECK: exit
  %r = call i64 @ext(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %a, i64 %b)
  %ab = add i64 %a, %b
  %cd = add i64 %c, %d
  %abcd = add i64 %ab, %cd
  %abcde = add i64 %abcd, %e
  %sum = add i64 %r, %abcde
  ret i64 %sum
}
