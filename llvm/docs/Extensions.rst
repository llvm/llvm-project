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

.. code-block:: gas

  .section .debug$S,"rn"
    .long 4
    .long 242
    .long 40
    .secrel32 _function_name
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

Target Specific Behaviour
=========================

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

