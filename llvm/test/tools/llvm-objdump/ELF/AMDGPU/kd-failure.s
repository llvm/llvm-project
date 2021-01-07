;; Failure test. We create a malformed kernel descriptor (KD) by manually
;; setting the bytes, because one can't create a malformed KD using the
;; assembler directives.

; RUN: llvm-mc %s --triple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj -o %t.o

; RUN: printf ".type  my_kernel.kd, @object \nmy_kernel.kd:\n.size my_kernel.kd, 64\n" > %t1.sym_info
; RUN: llvm-objdump --disassemble-symbols=my_kernel.kd %t.o \
; RUN: | tail -n +9 > %t1.sym_content
; RUN: cat %t1.sym_info %t1.sym_content > %t1.s

; RUN: llvm-mc %t1.s --triple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj -o %t-re-assemble.o
; RUN: diff %t.o %t-re-assemble.o

;; Test failure by setting one of the reserved bytes to non-zero value.

.type	my_kernel.kd, @object
.size my_kernel.kd, 64
my_kernel.kd:
  .long 0x00000000           ;; group_segment_fixed_size
  .long 0x00000000           ;; private_segment_fixed_size
  .long 0x00000000           ;; kernarg_segment_size.
  .long 0x00000000           ;; reserved bytes.
  .quad 0x0000000000000000   ;; kernel_code_entry_byte_offset, any value works.

  ;; 20 reserved bytes.
  .quad 0x00FF000000000000   ;; reserved bytes.
  .quad 0x0000000000000000
  .long 0x00000000

  .long 0x00000000           ;; compute_PGM_RSRC3
  .long 0x00000000           ;; compute_PGM_RSRC1
  .long 0x00000000           ;; compute_PGM_RSRC2
  .short 0x0000              ;; additional fields.

  ;; 6 reserved bytes.
  .long 0x0000000
  .short 0x0000
