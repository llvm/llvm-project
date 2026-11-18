; RUN: llc -mtriple=bpfel -mcpu=v4 -o - %s | FileCheck %s
; RUN: llc -mtriple=bpfel -mcpu=v4 -filetype=obj -o - %s \
; RUN:   | llvm-objdump -dr --no-show-raw-insn - | FileCheck --check-prefix=OBJ %s

; Block addresses and tables of them used by another function are
; data, not jump tables.

@tbl_owner = private constant [2 x ptr] [ptr blockaddress(@owner, %a), ptr blockaddress(@owner, %b)]
@tbl_shared = private constant [2 x ptr] [ptr blockaddress(@owner, %a), ptr blockaddress(@owner, %b)]
@tbl_mixed = private constant [2 x ptr] [ptr blockaddress(@owner, %a), ptr blockaddress(@other, %x)]

define void @user_first(ptr %p) {
  store volatile ptr blockaddress(@owner, %a), ptr %p
  store volatile ptr @tbl_owner, ptr %p
  store volatile ptr @tbl_shared, ptr %p
  ret void
}

define void @owner(i64 %i, ptr %p) {
entry:
  store volatile ptr @tbl_mixed, ptr %p
  %gep = getelementptr [2 x ptr], ptr @tbl_shared, i64 0, i64 %i
  %tgt = load ptr, ptr %gep
  indirectbr ptr %tgt, [label %a, label %b]

a:
  store volatile i32 1, ptr %p
  ret void

b:
  store volatile i32 2, ptr %p
  ret void
}

define void @user_last(ptr %p) {
  store volatile ptr blockaddress(@owner, %b), ptr %p
  ret void
}

define void @other(i1 %c, ptr %p) {
entry:
  br i1 %c, label %x, label %y

x:
  store volatile i32 3, ptr %p
  ret void

y:
  ret void
}

; CHECK-LABEL: user_first:
; CHECK:         r{{[0-9]}} = [[A:.Ltmp[0-9]+]] ll
; CHECK:         r{{[0-9]}} = .Ltbl_owner ll
; CHECK:         r{{[0-9]}} = .Ltbl_shared ll
; CHECK-LABEL: owner:
; CHECK:         r{{[0-9]}} = .Ltbl_mixed ll
; CHECK:         r{{[0-9]}} = BPF.JT.1.0 ll
; CHECK:         gotox r{{[0-9]}}
; CHECK-NEXT:  [[A]]: # Block address taken
; CHECK-NEXT:  .LBB1_1: # %a
; CHECK:       [[B:.Ltmp[0-9]+]]: # Block address taken
; CHECK-NEXT:  .LBB1_2: # %b
; CHECK:       BPF.JT.1.0:
; CHECK-NEXT:    .quad .LBB1_1-.text
; CHECK-NEXT:    .quad .LBB1_2-.text
; CHECK-LABEL: user_last:
; CHECK:         r{{[0-9]}} = [[B]] ll
; CHECK-LABEL: other:
; CHECK:       [[X:.Ltmp[0-9]+]]: # Block address taken
; CHECK:       .Ltbl_owner:
; CHECK-NEXT:    .quad [[A]]
; CHECK-NEXT:    .quad [[B]]
; CHECK:       .Ltbl_shared:
; CHECK-NEXT:    .quad [[A]]
; CHECK-NEXT:    .quad [[B]]
; CHECK:       .Ltbl_mixed:
; CHECK-NEXT:    .quad [[A]]
; CHECK-NEXT:    .quad [[X]]

; OBJ-LABEL: <user_first>:
; OBJ:         r{{[0-9]}} = 0x{{[0-9a-f]+}} ll
; OBJ-NEXT:      R_BPF_64_64 .text
; OBJ:         r{{[0-9]}} = 0x{{[0-9a-f]+}} ll
; OBJ-NEXT:      R_BPF_64_64 .data.rel.ro
; OBJ:         r{{[0-9]}} = 0x{{[0-9a-f]+}} ll
; OBJ-NEXT:      R_BPF_64_64 .data.rel.ro
; OBJ-LABEL: <owner>:
; OBJ:         r{{[0-9]}} = 0x{{[0-9a-f]+}} ll
; OBJ-NEXT:      R_BPF_64_64 .data.rel.ro
; OBJ:         r{{[0-9]}} = 0x0 ll
; OBJ-NEXT:      R_BPF_64_64 BPF.JT.1.0
; OBJ-LABEL: <user_last>:
; OBJ:         r{{[0-9]}} = 0x{{[0-9a-f]+}} ll
; OBJ-NEXT:      R_BPF_64_64 .text
