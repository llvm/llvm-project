;; RUN: llc -mtriple=hexagon-unknown-elf -filetype=obj %s -o - \
;; RUN: | llvm-readobj -h -r | FileCheck -check-prefix=OBJ %s

; OBJ: Format: elf32-hexagon
; OBJ: Arch: hexagon
; OBJ: AddressSize: 32bit
; OBJ: Machine: EM_HEXAGON (0xA4)
