OUTPUT_ARCH("i386:x86-64")

ENTRY(_start)

MEMORY
{
  RAM (xrw) : ORIGIN = 0x500000, LENGTH = 2048M
}

SECTIONS
{
  /* Define the memory layout */
  . = 0x500000;

  /* Text section */
  .text : {
    *(.text)
  } > RAM

  /* Data section */
  .data : {
    *(.data)
  } > RAM

  /* BSS section */
  .bss : {
    *(.bss)
  } > RAM

  /* Custom section*/
  .def_section : {
    *(.def_section)
  } > RAM
}
