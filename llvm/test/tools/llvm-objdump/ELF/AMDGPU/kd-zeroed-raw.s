; RUN: llvm-mc %s -mattr=+code-object-v3 --triple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj -o %t1
; RUN: llvm-objdump --disassemble-symbols=my_kernel.kd %t1 \
; RUN: | tail -n +8 | llvm-mc --triple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj -o %t2
; RUN: llvm-objdump -s -j .text %t2 | FileCheck --check-prefix=OBJDUMP %s

;; Not running lit-test over gfx10 (see kd-zeroed-gfx10.s for details).
;; kd-zeroed-raw.s and kd-zeroed-*.s should produce the same output for the
;; kernel descriptor - a block of 64 zeroed bytes.

;; The disassembly will produce the contents of kd-zeroed-*.s which on being
;; assembled contains additional relocation info. A diff over the entire object
;; will fail in this case. So we check by looking the bytes in .text.

; OBJDUMP:      0000 00000000 00000000 00000000 00000000
; OBJDUMP-NEXT: 0010 00000000 00000000 00000000 00000000
; OBJDUMP-NEXT: 0020 00000000 00000000 00000000 00000000
; OBJDUMP-NEXT: 0030 00000000 00000000 00000000 00000000

;; The entire object is zeroed out.

.type	my_kernel.kd, @object
.size my_kernel.kd, 64
my_kernel.kd:
  .long 0x00000000           ;; group_segment_fixed_size
  .long 0x00000000           ;; private_segment_fixed_size
  .quad 0x0000000000000000   ;; reserved bytes.
  .quad 0x0000000000000000   ;; kernel_code_entry_byte_offset, any value works.

  ;; 20 reserved bytes.
  .quad 0x0000000000000000
  .quad 0x0000000000000000
  .long 0x00000000

  .long 0x00000000           ;; compute_PGM_RSRC3
  .long 0x00000000           ;; compute_PGM_RSRC1
  .long 0x00000000           ;; compute_PGM_RSRC2
  .short 0x0000              ;; additional fields.

  ;; 6 reserved bytes.
  .long 0x0000000
  .short 0x0000
