; RUN: llc -mtriple=bpfel -mcpu=v4 -O0 -o - %s | FileCheck %s

; What follows the start of an unwind is dead and goes away, also a block whose
; address is taken: AsmPrinter still emits its label.

@addr = global ptr blockaddress(@address_taken, %taken)
@tbl = constant [3 x ptr] [ptr blockaddress(@table, %a), ptr blockaddress(@table, %b), ptr blockaddress(@table, %dead)]
@dead_tbl = constant [2 x ptr] [ptr blockaddress(@dead_table, %a), ptr blockaddress(@dead_table, %b)]
@other_tbl = constant [2 x ptr] [ptr blockaddress(@owner, %a), ptr blockaddress(@owner, %b)]

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
; CHECK-NEXT:  [[TAKEN:.Ltmp[0-9]+]]: # Address of block that was removed by CodeGen
; CHECK-NEXT:  .Lfunc_end{{[0-9]+}}:

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
; CHECK-NEXT:  .Lfunc_end{{[0-9]+}}:

; A computed goto through a table that also holds a block behind the unwind.
define void @table(i32 %x, ptr %p) uwtable {
entry:
  %c = icmp eq i32 %x, 7
  br i1 %c, label %fail, label %go

fail:
  call void @llvm.bpf.unwind.exit()
  br label %dead

go:
  %i = and i32 %x, 1
  %ix = zext i32 %i to i64
  %slot = getelementptr [3 x ptr], ptr @tbl, i64 0, i64 %ix
  %t = load ptr, ptr %slot
  indirectbr ptr %t, [label %a, label %b]

a:
  store volatile i32 1, ptr %p
  ret void

b:
  store volatile i32 2, ptr %p
  ret void

dead:
  store volatile i32 3, ptr %p
  ret void
}

; CHECK-LABEL: table:
; CHECK:       # %bb.1: # %fail
; CHECK-NEXT:    r0 = 0
; CHECK-NEXT:    unwind_exit
; CHECK:         gotox r1
; CHECK-NEXT:  [[TBL_A:.Ltmp[0-9]+]]: # Block address taken
; CHECK-NEXT:  .LBB{{[0-9_]+}}: # %a
; CHECK:       [[TBL_B:.Ltmp[0-9]+]]: # Block address taken
; CHECK-NEXT:  .LBB{{[0-9_]+}}: # %b
; CHECK:         exit
; CHECK-NEXT:  [[TBL_DEAD:.Ltmp[0-9]+]]: # Address of block that was removed by CodeGen
; CHECK-NEXT:  .Lfunc_end{{[0-9]+}}:

; The table and the computed goto are both behind the unwind.
define void @dead_table(i32 %x, ptr %p) uwtable {
entry:
  call void @llvm.bpf.unwind.exit()
  br label %go

go:
  %i = and i32 %x, 1
  %ix = zext i32 %i to i64
  %slot = getelementptr [2 x ptr], ptr @dead_tbl, i64 0, i64 %ix
  %t = load ptr, ptr %slot
  indirectbr ptr %t, [label %a, label %b]

a:
  store volatile i32 1, ptr %p
  ret void

b:
  store volatile i32 2, ptr %p
  ret void
}

; CHECK-LABEL: dead_table:
; CHECK:         r0 = 0
; CHECK-NEXT:    unwind_exit
; CHECK-NEXT:  [[DEAD_A:.Ltmp[0-9]+]]: # Address of block that was removed by CodeGen
; CHECK-NEXT:  [[DEAD_B:.Ltmp[0-9]+]]: # Address of block that was removed by CodeGen
; CHECK-NEXT:  .Lfunc_end{{[0-9]+}}:

; Another function reads a table of blocks that are behind the unwind.
define ptr @other_user(i64 %i) {
  %slot = getelementptr [2 x ptr], ptr @other_tbl, i64 0, i64 %i
  %t = load ptr, ptr %slot
  ret ptr %t
}

define void @owner(i32 %x, ptr %p) uwtable {
entry:
  call void @llvm.bpf.unwind.exit()
  br label %go

go:
  %c = icmp eq i32 %x, 0
  br i1 %c, label %a, label %b

a:
  store volatile i32 1, ptr %p
  ret void

b:
  store volatile i32 2, ptr %p
  ret void
}

; CHECK-LABEL: other_user:
; CHECK:         r1 = other_tbl ll
; CHECK-LABEL: owner:
; CHECK:         r0 = 0
; CHECK-NEXT:    unwind_exit
; CHECK-NEXT:  [[OTHER_A:.Ltmp[0-9]+]]: # Address of block that was removed by CodeGen
; CHECK-NEXT:  [[OTHER_B:.Ltmp[0-9]+]]: # Address of block that was removed by CodeGen
; CHECK-NEXT:  .Lfunc_end{{[0-9]+}}:

; The address is taken before the unwind, and only the indirectbr is behind it.
; FIXME: The address is loaded from a jump table that is never emitted.
define void @live_blockaddress(ptr %p) uwtable {
entry:
  store volatile ptr blockaddress(@live_blockaddress, %t), ptr %p
  call void @llvm.bpf.unwind.exit()
  br label %ib

ib:
  %a = load volatile ptr, ptr %p
  indirectbr ptr %a, [label %t]

t:
  store volatile i32 1, ptr %p
  ret void
}

; CHECK-LABEL: live_blockaddress:
; CHECK:         r1 = [[JT:BPF.JT.[0-9]+.[0-9]+]] ll
; CHECK-NEXT:    r1 = *(u64 *)(r1 + 0)
; CHECK-NEXT:    *(u64 *)(r2 + 0) = r1
; CHECK-NEXT:    r0 = 0
; CHECK-NEXT:    unwind_exit
; CHECK-NEXT:  .Ltmp{{[0-9]+}}: # Address of block that was removed by CodeGen
; CHECK-NEXT:  .Lfunc_end{{[0-9]+}}:
; CHECK:         .section .jumptables,"",@progbits
; CHECK-NEXT:  # -- End function
; CHECK-NEXT:    .text

; A switch lowered to a jump table, behind the unwind.
define void @dead_switch(i32 %x, ptr %p) uwtable {
entry:
  call void @llvm.bpf.unwind.exit()
  br label %sw

sw:
  switch i32 %x, label %done [
    i32 0, label %c0
    i32 1, label %c1
    i32 2, label %c2
    i32 3, label %c3
    i32 4, label %c4
    i32 5, label %c5
    i32 6, label %c6
    i32 7, label %c7
    i32 8, label %c8
    i32 9, label %c9
    i32 10, label %c10
    i32 11, label %c11
    i32 12, label %c12
    i32 13, label %c13
  ]

c0:
  store volatile i32 0, ptr %p
  br label %done

c1:
  store volatile i32 1, ptr %p
  br label %done

c2:
  store volatile i32 2, ptr %p
  br label %done

c3:
  store volatile i32 3, ptr %p
  br label %done

c4:
  store volatile i32 4, ptr %p
  br label %done

c5:
  store volatile i32 5, ptr %p
  br label %done

c6:
  store volatile i32 6, ptr %p
  br label %done

c7:
  store volatile i32 7, ptr %p
  br label %done

c8:
  store volatile i32 8, ptr %p
  br label %done

c9:
  store volatile i32 9, ptr %p
  br label %done

c10:
  store volatile i32 10, ptr %p
  br label %done

c11:
  store volatile i32 11, ptr %p
  br label %done

c12:
  store volatile i32 12, ptr %p
  br label %done

c13:
  store volatile i32 13, ptr %p
  br label %done

done:
  ret void
}

; CHECK-LABEL: dead_switch:
; CHECK:         r0 = 0
; CHECK-NEXT:    unwind_exit
; CHECK-NEXT:  .Lfunc_end{{[0-9]+}}:

; CHECK-LABEL: addr:
; CHECK-NEXT:    .quad [[TAKEN]]{{$}}
; CHECK-LABEL: tbl:
; CHECK-NEXT:    .quad [[TBL_A]]{{$}}
; CHECK-NEXT:    .quad [[TBL_B]]{{$}}
; CHECK-NEXT:    .quad [[TBL_DEAD]]{{$}}
; CHECK-LABEL: dead_tbl:
; CHECK-NEXT:    .quad [[DEAD_A]]{{$}}
; CHECK-NEXT:    .quad [[DEAD_B]]{{$}}
; CHECK-LABEL: other_tbl:
; CHECK-NEXT:    .quad [[OTHER_A]]{{$}}
; CHECK-NEXT:    .quad [[OTHER_B]]{{$}}
