; RUN: llc -mtriple=bpfel -mcpu=v4 -O0 -bpf-min-jump-table-entries=4 -o - %s | FileCheck %s

; What follows the start of an unwind is dead and goes away, except a block
; whose address is taken and what it branches to.

@addr = global ptr blockaddress(@address_taken, %taken)

define void @address_taken(ptr %p) uwtable {
entry:
  call void @llvm.bpf.unwind.exit()
  br label %taken

taken:
  store i32 2, ptr %p
  br label %join

join:
  store i32 3, ptr %p
  ret void
}

; CHECK-LABEL: address_taken:
; CHECK:         r0 = 0
; CHECK-NEXT:    unwind_exit
; CHECK-NEXT:  .Ltmp0: # Block address taken
; CHECK-NEXT:  # %bb.1: # %taken
; CHECK-NEXT:    r1 = *(u64 *)(r10 - 8)
; CHECK-NEXT:    *(u32 *)(r1 + 0) = 2
; CHECK-NEXT:  # %bb.2: # %join
; CHECK-NEXT:    r1 = *(u64 *)(r10 - 8)
; CHECK-NEXT:    *(u32 *)(r1 + 0) = 3
; CHECK-NEXT:    exit

; A dead block can be the successor of dead blocks after it.
define void @dead_loop(ptr %p) uwtable {
entry:
  call void @llvm.bpf.unwind.exit()
  br label %loop

loop:
  %v = load volatile i32, ptr %p
  %c = icmp eq i32 %v, 0
  br i1 %c, label %loop, label %a

a:
  %va = load volatile i32, ptr %p
  %ca = icmp eq i32 %va, 0
  br i1 %ca, label %loop, label %b

b:
  %vb = load volatile i32, ptr %p
  %cb = icmp eq i32 %vb, 0
  br i1 %cb, label %loop, label %d

d:
  %vd = load volatile i32, ptr %p
  %cd = icmp eq i32 %vd, 0
  br i1 %cd, label %loop, label %done

done:
  ret void
}

; CHECK-LABEL: dead_loop:
; CHECK:         r0 = 0
; CHECK-NEXT:    unwind_exit
; CHECK-NEXT:  .Lfunc_end1:

; The jump table of a dead switch loses its entries.
define void @dead_switch(ptr %p, i32 %x) uwtable {
entry:
  call void @llvm.bpf.unwind.exit()
  switch i32 %x, label %done [
    i32 0, label %a
    i32 1, label %b
    i32 2, label %c
    i32 3, label %d
  ]

a:
  store volatile i32 0, ptr %p
  br label %done

b:
  store volatile i32 1, ptr %p
  br label %done

c:
  store volatile i32 2, ptr %p
  br label %done

d:
  store volatile i32 3, ptr %p
  br label %done

done:
  ret void
}

; CHECK-LABEL: dead_switch:
; CHECK:         r0 = 0
; CHECK-NEXT:    unwind_exit
; CHECK-NEXT:  .Lfunc_end2:
; CHECK-NEXT:    .size dead_switch, .Lfunc_end2-dead_switch
; CHECK-NEXT:    .cfi_endproc
; CHECK-NEXT:    .section .jumptables,"",@progbits
; CHECK-NEXT:                                          # -- End function
