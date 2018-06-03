===============
LLVM Extensions
===============

.. contents::
   :local:

.. toctree::
   :hidden:

Introduction
============

This document describes extensions to tools and formats LLVM seeks compatibility
with.

General Assembly Syntax
===========================

C99-style Hexadecimal Floating-point Constants
----------------------------------------------

LLVM's assemblers allow floating-point constants to be written in C99's
hexadecimal format instead of decimal if desired.

.. code-block:: gas

  .section .data
  .float 0x1c2.2ap3

Machine-specific Assembly Syntax
================================

X86/COFF-Dependent
------------------

Relocations
^^^^^^^^^^^

The following additional relocation types are supported:

**@IMGREL** (AT&T syntax only) generates an image-relative relocation that
corresponds to the COFF relocation types ``IMAGE_REL_I386_DIR32NB`` (32-bit) or
``IMAGE_REL_AMD64_ADDR32NB`` (64-bit).

.. code-block:: text

  .text
  fun:
    mov foo@IMGREL(%ebx, %ecx, 4), %eax

  .section .pdata
    .long fun@IMGREL
    .long (fun@imgrel + 0x3F)
    .long $unwind$fun@imgrel

**.secrel32** generates a relocation that corresponds to the COFF relocation
types ``IMAGE_REL_I386_SECREL`` (32-bit) or ``IMAGE_REL_AMD64_SECREL`` (64-bit).

**.secidx** relocation generates an index of the section that contains
the target.  It corresponds to the COFF relocation types
``IMAGE_REL_I386_SECTION`` (32-bit) or ``IMAGE_REL_AMD64_SECTION`` (64-bit).

.. code-block:: none

  .section .debug$S,"rn"
    .long 4
    .long 242
    .long 40
    .secrel32 _function_name + 0
    .secidx   _function_name
    ...

``.linkonce`` Directive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:

   ``.linkonce [ comdat type ]``

Supported COMDAT types:

``discard``
   Discards duplicate sections with the same COMDAT symbol. This is the default
   if no type is specified.

``one_only``
   If the symbol is defined multiple times, the linker issues an error.

``same_size``
   Duplicates are discarded, but the linker issues an error if any have
   different sizes.

``same_contents``
   Duplicates are discarded, but the linker issues an error if any duplicates
   do not have exactly the same content.

``largest``
   Links the largest section from among the duplicates.

``newest``
   Links the newest section from among the duplicates.


.. code-block:: gas

  .section .text$foo
  .linkonce
    ...

``.section`` Directive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MC supports passing the information in ``.linkonce`` at the end of
``.section``. For example,  these two codes are equivalent

.. code-block:: gas

  .section secName, "dr", discard, "Symbol1"
  .globl Symbol1
  Symbol1:
  .long 1

.. code-block:: gas

  .section secName, "dr"
  .linkonce discard
  .globl Symbol1
  Symbol1:
  .long 1

Note that in the combined form the COMDAT symbol is explicit. This
extension exists to support multiple sections with the same name in
different COMDATs:


.. code-block:: gas

  .section secName, "dr", discard, "Symbol1"
  .globl Symbol1
  Symbol1:
  .long 1

  .section secName, "dr", discard, "Symbol2"
  .globl Symbol2
  Symbol2:
  .long 1

In addition to the types allowed with ``.linkonce``, ``.section`` also accepts
``associative``. The meaning is that the section is linked  if a certain other
COMDAT section is linked. This other section is indicated by the comdat symbol
in this directive. It can be any symbol defined in the associated section, but
is usually the associated section's comdat.

   The following restrictions apply to the associated section:

   1. It must be a COMDAT section.
   2. It cannot be another associative COMDAT section.

In the following example the symobl ``sym`` is the comdat symbol of ``.foo``
and ``.bar`` is associated to ``.foo``.

.. code-block:: gas

	.section	.foo,"bw",discard, "sym"
	.section	.bar,"rd",associative, "sym"

MC supports these flags in the COFF ``.section`` directive:

  - ``b``: BSS section (``IMAGE_SCN_CNT_INITIALIZED_DATA``)
  - ``d``: Data section (``IMAGE_SCN_CNT_UNINITIALIZED_DATA``)
  - ``n``: Section is not loaded (``IMAGE_SCN_LNK_REMOVE``)
  - ``r``: Read-only
  - ``s``: Shared section
  - ``w``: Writable
  - ``x``: Executable section
  - ``y``: Not readable
  - ``D``: Discardable (``IMAGE_SCN_MEM_DISCARDABLE``)

These flags are all compatible with gas, with the exception of the ``D`` flag,
which gnu as does not support. For gas compatibility, sections with a name
starting with ".debug" are implicitly discardable.


ARM64/COFF-Dependent
--------------------

Relocations
^^^^^^^^^^^

The following additional symbol variants are supported:

**:secrel_lo12:** generates a relocation that corresponds to the COFF relocation
types ``IMAGE_REL_ARM64_SECREL_LOW12A`` or ``IMAGE_REL_ARM64_SECREL_LOW12L``.

**:secrel_hi12:** generates a relocation that corresponds to the COFF relocation
type ``IMAGE_REL_ARM64_SECREL_HIGH12A``.

.. code-block:: gas

    add x0, x0, :secrel_hi12:symbol
    ldr x0, [x0, :secrel_lo12:symbol]

    add x1, x1, :secrel_hi12:symbol
    add x1, x1, :secrel_lo12:symbol
    ...


ELF-Dependent
-------------

``.section`` Directive
^^^^^^^^^^^^^^^^^^^^^^

In order to support creating multiple sections with the same name and comdat,
it is possible to add an unique number at the end of the ``.seciton`` directive.
For example, the following code creates two sections named ``.text``.

.. code-block:: gas

	.section	.text,"ax",@progbits,unique,1
        nop

	.section	.text,"ax",@progbits,unique,2
        nop


The unique number is not present in the resulting object at all. It is just used
in the assembler to differentiate the sections.

The 'o' flag is mapped to SHF_LINK_ORDER. If it is present, a symbol
must be given that identifies the section to be placed is the
.sh_link.

.. code-block:: gas

        .section .foo,"a",@progbits
        .Ltmp:
        .section .bar,"ao",@progbits,.Ltmp

which is equivalent to just

.. code-block:: gas

        .section .foo,"a",@progbits
        .section .bar,"ao",@progbits,.foo

``.linker-options`` Section (linker options)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to support passing linker options from the frontend to the linker, a
special section of type ``SHT_LLVM_LINKER_OPTIONS`` (usually named
``.linker-options`` though the name is not significant as it is identified by
the type).  The contents of this section is a simple pair-wise encoding of
directives for consideration by the linker.  The strings are encoded as standard
null-terminated UTF-8 strings.  They are emitted inline to avoid having the
linker traverse the object file for retrieving the value.  The linker is
permitted to not honour the option and instead provide a warning/error to the
user that the requested option was not honoured.

The section has type ``SHT_LLVM_LINKER_OPTIONS`` and has the ``SHF_EXCLUDE``
flag to ensure that the section is treated as opaque by linkers which do not
support the feature and will not be emitted into the final linked binary.

This would be equivalent to the follow raw assembly:

.. code-block:: gas

  .section ".linker-options","e",@llvm_linker_options
  .asciz "option 1"
  .asciz "value 1"
  .asciz "option 2"
  .asciz "value 2"

The following directives are specified:

  - lib

    The parameter identifies a library to be linked against.  The library will
    be looked up in the default and any specified library search paths
    (specified to this point).

  - libpath

    The paramter identifies an additional library search path to be considered
    when looking up libraries after the inclusion of this option.

``SHT_LLVM_CALL_GRAPH_PROFILE`` Section (Call Graph Profile)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section is used to pass a call graph profile to the linker which can be
used to optimize the placement of sections.  It contains a sequence of
(from symbol, to symbol, weight) tuples.

It shall have a type of ``SHT_LLVM_CALL_GRAPH_PROFILE`` (0x6fff4c02), shall
have the ``SHF_EXCLUDE`` flag set, the ``sh_link`` member shall hold the section
header index of the associated symbol table, and shall have a ``sh_entsize`` of
16.  It should be named ``.llvm.call-graph-profile``.

The contents of the section shall be a sequence of ``Elf_CGProfile`` entries.

.. code-block:: c

  typedef struct {
    Elf_Word cgp_from;
    Elf_Word cgp_to;
    Elf_Xword cgp_weight;
  } Elf_CGProfile;

cgp_from
  The symbol index of the source of the edge.

cgp_to
  The symbol index of the destination of the edge.

cgp_weight
  The weight of the edge.

This is represented in assembly as:

.. code-block:: gas

  .cg_profile from, to, 42

``.cg_profile`` directives are processed at the end of the file.  It is an error
if either ``from`` or ``to`` are undefined temporary symbols.  If either symbol
is a temporary symbol, then the section symbol is used instead.  If either
symbol is undefined, then that symbol is defined as if ``.weak symbol`` has been
written at the end of the file.  This forces the symbol to show up in the symbol
table.

Target Specific Behaviour
=========================

X86
---

Relocations
^^^^^^^^^^^

``@ABS8`` can be applied to symbols which appear as immediate operands to
instructions that have an 8-bit immediate form for that operand. It causes
the assembler to use the 8-bit form and an 8-bit relocation (e.g. ``R_386_8``
or ``R_X86_64_8``) for the symbol.

For example:

.. code-block:: gas

  cmpq $foo@ABS8, %rdi

This causes the assembler to select the form of the 64-bit ``cmpq`` instruction
that takes an 8-bit immediate operand that is sign extended to 64 bits, as
opposed to ``cmpq $foo, %rdi`` which takes a 32-bit immediate operand. This
is also not the same as ``cmpb $foo, %dil``, which is an 8-bit comparison.

Windows on ARM
--------------

Stack Probe Emission
^^^^^^^^^^^^^^^^^^^^

The reference implementation (Microsoft Visual Studio 2012) emits stack probes
in the following fashion:

.. code-block:: gas

  movw r4, #constant
  bl __chkstk
  sub.w sp, sp, r4

However, this has the limitation of 32 MiB (±16MiB).  In order to accommodate
larger binaries, LLVM supports the use of ``-mcode-model=large`` to allow a 4GiB
range via a slight deviation.  It will generate an indirect jump as follows:

.. code-block:: gas

  movw r4, #constant
  movw r12, :lower16:__chkstk
  movt r12, :upper16:__chkstk
  blx r12
  sub.w sp, sp, r4

Variable Length Arrays
^^^^^^^^^^^^^^^^^^^^^^

The reference implementation (Microsoft Visual Studio 2012) does not permit the
emission of Variable Length Arrays (VLAs).

The Windows ARM Itanium ABI extends the base ABI by adding support for emitting
a dynamic stack allocation.  When emitting a variable stack allocation, a call
to ``__chkstk`` is emitted unconditionally to ensure that guard pages are setup
properly.  The emission of this stack probe emission is handled similar to the
standard stack probe emission.

The MSVC environment does not emit code for VLAs currently.

Windows on ARM64
----------------

Stack Probe Emission
^^^^^^^^^^^^^^^^^^^^

The reference implementation (Microsoft Visual Studio 2017) emits stack probes
in the following fashion:

.. code-block:: gas

  mov x15, #constant
  bl __chkstk
  sub sp, sp, x15, lsl #4

However, this has the limitation of 256 MiB (±128MiB).  In order to accommodate
larger binaries, LLVM supports the use of ``-mcode-model=large`` to allow a 8GiB
(±4GiB) range via a slight deviation.  It will generate an indirect jump as
follows:

.. code-block:: gas

  mov x15, #constant
  adrp x16, __chkstk
  add x16, x16, :lo12:__chkstk
  blr x16
  sub sp, sp, x15, lsl #4

