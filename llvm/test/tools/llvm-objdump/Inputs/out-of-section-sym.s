// $ cat out-of-section-sym.ld
// SECTIONS
// {
//   . = 0x10;
//   .text : { _ftext = . ; *(.text) }
//   . = 0x20;
//   .data : { _fdata = . ; *(.data) }
// }
// as --32 out-of-section-sym.s -o out-of-section-sym.o
// ld -m elf_i386 -Tout-of-section-sym.ld -o out-of-section-sym.elf-i386 \
//    out-of-section-sym.o

.text
_start:
  ret
