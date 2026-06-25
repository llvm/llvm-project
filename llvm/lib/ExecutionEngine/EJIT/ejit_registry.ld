/*===-- ejit_registry.ld - EmbeddedJIT registry linker script (example) --===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions. See https://llvm.org/LICENSE.txt for license information.      *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/*
 * EXAMPLE linker-script fragment for bare-metal targets whose linker does NOT
 * auto-synthesize the __start_/__stop_ bound symbols for custom sections.
 *
 * Background
 * ----------
 * PASS1 (EJitRegisterBitcode) and PASS2 (EJitRegisterPeriod) emit each
 * translation unit's registry entries into INPUT sections named:
 *     .ejit_bitcode   (array of ejit_reg_entry_t: embedded bitcode + symbols)
 *     .ejit_period    (array of ejit_reg_entry_t: period arrays + static vars)
 * Each entry is `struct { i32 type; ptr; ptr; ptr; u64; }` -> 8-byte aligned,
 * 40 bytes on a 64-bit target. Every entry array is private (local) so there
 * is no cross-TU symbol collision; the entries from all TUs must simply end up
 * contiguous so the runtime can iterate them.
 *
 * The runtime (EJit.cpp) walks the half-open range
 *     [__start_ejit_bitcode, __stop_ejit_bitcode)
 *     [__start_ejit_period,  __stop_ejit_period)
 * These four symbols are declared `weak` in the runtime, so if they are left
 * undefined (no ejit_entry functions, or no script) they resolve to 0 and the
 * range is empty. The section names carry a leading dot (.ejit_bitcode), which
 * is the conventional ELF spelling but is NOT a valid C identifier, so the
 * linker never auto-synthesizes __start_/__stop_ for them — this script (or an
 * equivalent in your own linker script) is REQUIRED to define the bounds.
 *
 * What this script does
 * ---------------------
 * It folds the .ejit_bitcode / .ejit_period input sections into a STANDARD
 * output section (here .rodata, since the tables are const) and brackets them
 * with the four bound symbols. KEEP() prevents --gc-sections from dropping the
 * tables (they are only referenced by linker-defined symbols, never by a
 * normal relocation, so they look "unused" to the garbage collector).
 *
 * Integration options
 * -------------------
 *   (A) Paste the two blocks below into your existing .rodata (or .data)
 *       output-section body, between your normal `*(.rodata*)` lines.
 *   (B) Keep them in a separate file and pull it in with INSERT (see the
 *       INSERT variant at the bottom) so you do not have to edit the vendor
 *       script's SECTIONS command.
 *
 * Placement note (.rodata vs .data)
 * ---------------------------------
 * The tables are emitted `constant`; in a static (non-PIC) bare-metal image
 * their pointer fields are resolved to absolute addresses at link time, so
 * they are genuinely read-only and belong in .rodata / flash — no RAM cost.
 * If your layout has no separate .rodata, the identical blocks work inside
 * .data (they will then be copied to RAM by your startup .data init loop).
 */

/* ----------------------------------------------------------------------- */
/* (A) Inline form — drop these two blocks inside an output section body.   */
/*     Shown here inside a complete (skeletal) SECTIONS for clarity.        */
/* ----------------------------------------------------------------------- */

SECTIONS
{
  /* ... your existing .text / .vectors / etc. ... */

  .rodata : ALIGN(8)
  {
    *(.rodata .rodata.*)

    /* --- EmbeddedJIT bitcode registry --- */
    . = ALIGN(8);
    __start_ejit_bitcode = .;
    KEEP(*(.ejit_bitcode))
    __stop_ejit_bitcode = .;

    /* --- EmbeddedJIT period registry --- */
    . = ALIGN(8);
    __start_ejit_period = .;
    KEEP(*(.ejit_period))
    __stop_ejit_period = .;
    . = ALIGN(8);
  } > FLASH      /* or your rodata memory region; use > RAM AT> FLASH for .data */

  /* ... your existing .data / .bss / etc. ... */
}

/*
 * ----------------------------------------------------------------------- *
 * (B) INSERT form — if you must NOT edit the vendor SECTIONS command.       *
 *     Put ONLY the snippet below in a separate script and pass it after the *
 *     main script. `INSERT AFTER .rodata` (or .data) merges it in without   *
 *     overriding the existing layout. Supported by both GNU ld and ld.lld.  *
 * ----------------------------------------------------------------------- *
 *
 *   SECTIONS
 *   {
 *     .ejit_registry : ALIGN(8)
 *     {
 *       . = ALIGN(8);
 *       __start_ejit_bitcode = .;
 *       KEEP(*(.ejit_bitcode))
 *       __stop_ejit_bitcode = .;
 *
 *       . = ALIGN(8);
 *       __start_ejit_period = .;
 *       KEEP(*(.ejit_period))
 *       __stop_ejit_period = .;
 *       . = ALIGN(8);
 *     }
 *   } INSERT AFTER .rodata;
 *
 * With the INSERT form the linker still places .ejit_registry next to .rodata
 * in the same loadable segment, so it does not become a "custom segment" — it
 * rides in the normal data/rodata program header.
 */
