=====================================
Syntax of AMDGPU Instruction Operands
=====================================

.. contents::
   :local:

Conventions
===========

The following notation is used throughout this document:

    =================== =============================================================================
    Notation            Description
    =================== =============================================================================
    {0..N}              Any integer value in the range from 0 to N (inclusive).
    <x>                 Syntax and meaning of *x* is explained elsewhere.
    =================== =============================================================================

.. _amdgpu_syn_operands:

Operands
========

.. _amdgpu_synid_v:

v
-

Vector registers. There are 256 32-bit vector registers.

A sequence of *vector* registers may be used to operate with more than 32 bits of data.

Assembler currently supports sequences of 1, 2, 3, 4, 8 and 16 *vector* registers.

    =================================================== ====================================================================
    Syntax                                              Description
    =================================================== ====================================================================
    **v**\<N>                                           A single 32-bit *vector* register.

                                                        *N* must be a decimal integer number.
    **v[**\ <N>\ **]**                                  A single 32-bit *vector* register.

                                                        *N* may be specified as an
                                                        :ref:`integer number<amdgpu_synid_integer_number>`
                                                        or an :ref:`absolute expression<amdgpu_synid_absolute_expression>`.
    **v[**\ <N>:<K>\ **]**                              A sequence of (\ *K-N+1*\ ) *vector* registers.

                                                        *N* and *K* may be specified as
                                                        :ref:`integer numbers<amdgpu_synid_integer_number>`
                                                        or :ref:`absolute expressions<amdgpu_synid_absolute_expression>`.
    **[v**\ <N>, \ **v**\ <N+1>, ... **v**\ <K>\ **]**  A sequence of (\ *K-N+1*\ ) *vector* registers.

                                                        Register indices must be specified as decimal integer numbers.
    =================================================== ====================================================================

Note. *N* and *K* must satisfy the following conditions:

* *N* <= *K*.
* 0 <= *N* <= 255.
* 0 <= *K* <= 255.
* *K-N+1* must be equal to 1, 2, 3, 4, 8 or 16.

Examples:

.. parsed-literal::

  v255
  v[0]
  v[0:1]
  v[1:1]
  v[0:3]
  v[2*2]
  v[1-1:2-1]
  [v252]
  [v252,v253,v254,v255]

.. _amdgpu_synid_nsa:

*Image* instructions may use special *NSA* (Non-Sequential Address) syntax for *image addresses*:

    =================================================== ====================================================================
    Syntax                                              Description
    =================================================== ====================================================================
    **[v**\ <A>, \ **v**\ <B>, ... **v**\ <X>\ **]**    A sequence of *vector* registers. At least one register
                                                        must be specified.

                                                        In contrast with standard syntax described above, registers in
                                                        this sequence are not required to have consecutive indices.
                                                        Moreover, the same register may appear in the list more than once.
    =================================================== ====================================================================

Note. Reqister indices must be in the range 0..255. They must be specified as decimal integer numbers.

Examples:

.. parsed-literal::

  [v32,v1,v2]
  [v4,v4,v4,v4]

.. _amdgpu_synid_s:

s
-

Scalar 32-bit registers. The number of available *scalar* registers depends on GPU:

    ======= ============================
    GPU     Number of *scalar* registers
    ======= ============================
    GFX7    104
    GFX8    102
    GFX9    102
    GFX10   106
    ======= ============================

A sequence of *scalar* registers may be used to operate with more than 32 bits of data.
Assembler currently supports sequences of 1, 2, 4, 8 and 16 *scalar* registers.

Pairs of *scalar* registers must be even-aligned (the first register must be even).
Sequences of 4 and more *scalar* registers must be quad-aligned.

    ======================================================== ====================================================================
    Syntax                                                   Description
    ======================================================== ====================================================================
    **s**\ <N>                                               A single 32-bit *scalar* register.

                                                             *N* must be a decimal integer number.
    **s[**\ <N>\ **]**                                       A single 32-bit *scalar* register.

                                                             *N* may be specified as an
                                                             :ref:`integer number<amdgpu_synid_integer_number>`
                                                             or an :ref:`absolute expression<amdgpu_synid_absolute_expression>`.
    **s[**\ <N>:<K>\ **]**                                   A sequence of (\ *K-N+1*\ ) *scalar* registers.

                                                             *N* and *K* may be specified as
                                                             :ref:`integer numbers<amdgpu_synid_integer_number>`
                                                             or :ref:`absolute expressions<amdgpu_synid_absolute_expression>`.
    **[s**\ <N>, \ **s**\ <N+1>, ... **s**\ <K>\ **]**       A sequence of (\ *K-N+1*\ ) *scalar* registers.

                                                             Register indices must be specified as decimal integer numbers.
    ======================================================== ====================================================================

Note. *N* and *K* must satisfy the following conditions:

* *N* must be properly aligned based on sequence size.
* *N* <= *K*.
* 0 <= *N* < *SMAX*\ , where *SMAX* is the number of available *scalar* registers.
* 0 <= *K* < *SMAX*\ , where *SMAX* is the number of available *scalar* registers.
* *K-N+1* must be equal to 1, 2, 4, 8 or 16.

Examples:

.. parsed-literal::

  s0
  s[0]
  s[0:1]
  s[1:1]
  s[0:3]
  s[2*2]
  s[1-1:2-1]
  [s4]
  [s4,s5,s6,s7]

Examples of *scalar* registers with an invalid alignment:

.. parsed-literal::

  s[1:2]
  s[2:5]

.. _amdgpu_synid_trap:

trap
----

A set of trap handler registers:

* :ref:`ttmp<amdgpu_synid_ttmp>`
* :ref:`tba<amdgpu_synid_tba>`
* :ref:`tma<amdgpu_synid_tma>`

.. _amdgpu_synid_ttmp:

ttmp
----

Trap handler temporary scalar registers, 32-bits wide.
The number of available *ttmp* registers depends on GPU:

    ======= ===========================
    GPU     Number of *ttmp* registers
    ======= ===========================
    GFX7    12
    GFX8    12
    GFX9    16
    GFX10   16
    ======= ===========================

A sequence of *ttmp* registers may be used to operate with more than 32 bits of data.
Assembler currently supports sequences of 1, 2, 4, 8 and 16 *ttmp* registers.

Pairs of *ttmp* registers must be even-aligned (the first register must be even).
Sequences of 4 and more *ttmp* registers must be quad-aligned.

    ============================================================= ====================================================================
    Syntax                                                        Description
    ============================================================= ====================================================================
    **ttmp**\ <N>                                                 A single 32-bit *ttmp* register.

                                                                  *N* must be a decimal integer number.
    **ttmp[**\ <N>\ **]**                                         A single 32-bit *ttmp* register.

                                                                  *N* may be specified as an
                                                                  :ref:`integer number<amdgpu_synid_integer_number>`
                                                                  or an :ref:`absolute expression<amdgpu_synid_absolute_expression>`.
    **ttmp[**\ <N>:<K>\ **]**                                     A sequence of (\ *K-N+1*\ ) *ttmp* registers.

                                                                  *N* and *K* may be specified as
                                                                  :ref:`integer numbers<amdgpu_synid_integer_number>`
                                                                  or :ref:`absolute expressions<amdgpu_synid_absolute_expression>`.
    **[ttmp**\ <N>, \ **ttmp**\ <N+1>, ... **ttmp**\ <K>\ **]**   A sequence of (\ *K-N+1*\ ) *ttmp* registers.

                                                                  Register indices must be specified as decimal integer numbers.
    ============================================================= ====================================================================

Note. *N* and *K* must satisfy the following conditions:

* *N* must be properly aligned based on sequence size.
* *N* <= *K*.
* 0 <= *N* < *TMAX*, where *TMAX* is the number of available *ttmp* registers.
* 0 <= *K* < *TMAX*, where *TMAX* is the number of available *ttmp* registers.
* *K-N+1* must be equal to 1, 2, 4, 8 or 16.

Examples:

.. parsed-literal::

  ttmp0
  ttmp[0]
  ttmp[0:1]
  ttmp[1:1]
  ttmp[0:3]
  ttmp[2*2]
  ttmp[1-1:2-1]
  [ttmp4]
  [ttmp4,ttmp5,ttmp6,ttmp7]

Examples of *ttmp* registers with an invalid alignment:

.. parsed-literal::

  ttmp[1:2]
  ttmp[2:5]

.. _amdgpu_synid_tba:

tba
---

Trap base address, 64-bits wide. Holds the pointer to the current trap handler program.

    ================== ======================================================================= =============
    Syntax             Description                                                             Availability
    ================== ======================================================================= =============
    tba                64-bit *trap base address* register.                                    GFX7, GFX8
    [tba]              64-bit *trap base address* register (an alternative syntax).            GFX7, GFX8
    [tba_lo,tba_hi]    64-bit *trap base address* register (an alternative syntax).            GFX7, GFX8
    ================== ======================================================================= =============

High and low 32 bits of *trap base address* may be accessed as separate registers:

    ================== ======================================================================= =============
    Syntax             Description                                                             Availability
    ================== ======================================================================= =============
    tba_lo             Low 32 bits of *trap base address* register.                            GFX7, GFX8
    tba_hi             High 32 bits of *trap base address* register.                           GFX7, GFX8
    [tba_lo]           Low 32 bits of *trap base address* register (an alternative syntax).    GFX7, GFX8
    [tba_hi]           High 32 bits of *trap base address* register (an alternative syntax).   GFX7, GFX8
    ================== ======================================================================= =============

Note that *tba*, *tba_lo* and *tba_hi* are not accessible as assembler registers in GFX9 and GFX10,
but *tba* is readable/writable with the help of *s_get_reg* and *s_set_reg* instructions.

.. _amdgpu_synid_tma:

tma
---

Trap memory address, 64-bits wide.

    ================= ======================================================================= ==================
    Syntax            Description                                                             Availability
    ================= ======================================================================= ==================
    tma               64-bit *trap memory address* register.                                  GFX7, GFX8
    [tma]             64-bit *trap memory address* register (an alternative syntax).          GFX7, GFX8
    [tma_lo,tma_hi]   64-bit *trap memory address* register (an alternative syntax).          GFX7, GFX8
    ================= ======================================================================= ==================

High and low 32 bits of *trap memory address* may be accessed as separate registers:

    ================= ======================================================================= ==================
    Syntax            Description                                                             Availability
    ================= ======================================================================= ==================
    tma_lo            Low 32 bits of *trap memory address* register.                          GFX7, GFX8
    tma_hi            High 32 bits of *trap memory address* register.                         GFX7, GFX8
    [tma_lo]          Low 32 bits of *trap memory address* register (an alternative syntax).  GFX7, GFX8
    [tma_hi]          High 32 bits of *trap memory address* register (an alternative syntax). GFX7, GFX8
    ================= ======================================================================= ==================

Note that *tma*, *tma_lo* and *tma_hi* are not accessible as assembler registers in GFX9 and GFX10,
but *tma* is readable/writable with the help of *s_get_reg* and *s_set_reg* instructions.

.. _amdgpu_synid_flat_scratch:

flat_scratch
------------

Flat scratch address, 64-bits wide. Holds the base address of scratch memory.

    ================================== ================================================================
    Syntax                             Description
    ================================== ================================================================
    flat_scratch                       64-bit *flat scratch* address register.
    [flat_scratch]                     64-bit *flat scratch* address register (an alternative syntax).
    [flat_scratch_lo,flat_scratch_hi]  64-bit *flat scratch* address register (an alternative syntax).
    ================================== ================================================================

High and low 32 bits of *flat scratch* address may be accessed as separate registers:

    ========================= =========================================================================
    Syntax                    Description
    ========================= =========================================================================
    flat_scratch_lo           Low 32 bits of *flat scratch* address register.
    flat_scratch_hi           High 32 bits of *flat scratch* address register.
    [flat_scratch_lo]         Low 32 bits of *flat scratch* address register (an alternative syntax).
    [flat_scratch_hi]         High 32 bits of *flat scratch* address register (an alternative syntax).
    ========================= =========================================================================

.. _amdgpu_synid_xnack:

xnack
-----

Xnack mask, 64-bits wide. Holds a 64-bit mask of which threads
received an *XNACK* due to a vector memory operation.

.. WARNING:: GFX7 does not support *xnack* feature. For availability of this feature in other GPUs, refer :ref:`this table<amdgpu-processors>`.

\

    ============================== =====================================================
    Syntax                         Description
    ============================== =====================================================
    xnack_mask                     64-bit *xnack mask* register.
    [xnack_mask]                   64-bit *xnack mask* register (an alternative syntax).
    [xnack_mask_lo,xnack_mask_hi]  64-bit *xnack mask* register (an alternative syntax).
    ============================== =====================================================

High and low 32 bits of *xnack mask* may be accessed as separate registers:

    ===================== ==============================================================
    Syntax                Description
    ===================== ==============================================================
    xnack_mask_lo         Low 32 bits of *xnack mask* register.
    xnack_mask_hi         High 32 bits of *xnack mask* register.
    [xnack_mask_lo]       Low 32 bits of *xnack mask* register (an alternative syntax).
    [xnack_mask_hi]       High 32 bits of *xnack mask* register (an alternative syntax).
    ===================== ==============================================================

.. _amdgpu_synid_vcc:
.. _amdgpu_synid_vcc_lo:

vcc
---

Vector condition code, 64-bits wide. A bit mask with one bit per thread;
it holds the result of a vector compare operation.

Note that GFX10 H/W does not use high 32 bits of *vcc* in *wave32* mode.

    ================ =========================================================================
    Syntax           Description
    ================ =========================================================================
    vcc              64-bit *vector condition code* register.
    [vcc]            64-bit *vector condition code* register (an alternative syntax).
    [vcc_lo,vcc_hi]  64-bit *vector condition code* register (an alternative syntax).
    ================ =========================================================================

High and low 32 bits of *vector condition code* may be accessed as separate registers:

    ================ =========================================================================
    Syntax           Description
    ================ =========================================================================
    vcc_lo           Low 32 bits of *vector condition code* register.
    vcc_hi           High 32 bits of *vector condition code* register.
    [vcc_lo]         Low 32 bits of *vector condition code* register (an alternative syntax).
    [vcc_hi]         High 32 bits of *vector condition code* register (an alternative syntax).
    ================ =========================================================================

.. _amdgpu_synid_m0:

m0
--

A 32-bit memory register. It has various uses,
including register indexing and bounds checking.

    =========== ===================================================
    Syntax      Description
    =========== ===================================================
    m0          A 32-bit *memory* register.
    [m0]        A 32-bit *memory* register (an alternative syntax).
    =========== ===================================================

.. _amdgpu_synid_exec:

exec
----

Execute mask, 64-bits wide. A bit mask with one bit per thread,
which is applied to vector instructions and controls which threads execute
and which ignore the instruction.

Note that GFX10 H/W does not use high 32 bits of *exec* in *wave32* mode.

    ===================== =================================================================
    Syntax                Description
    ===================== =================================================================
    exec                  64-bit *execute mask* register.
    [exec]                64-bit *execute mask* register (an alternative syntax).
    [exec_lo,exec_hi]     64-bit *execute mask* register (an alternative syntax).
    ===================== =================================================================

High and low 32 bits of *execute mask* may be accessed as separate registers:

    ===================== =================================================================
    Syntax                Description
    ===================== =================================================================
    exec_lo               Low 32 bits of *execute mask* register.
    exec_hi               High 32 bits of *execute mask* register.
    [exec_lo]             Low 32 bits of *execute mask* register (an alternative syntax).
    [exec_hi]             High 32 bits of *execute mask* register (an alternative syntax).
    ===================== =================================================================

.. _amdgpu_synid_vccz:

vccz
----

A single bit flag indicating that the :ref:`vcc<amdgpu_synid_vcc>` is all zeros.

Note. When GFX10 operates in *wave32* mode, this register reflects state of :ref:`vcc_lo<amdgpu_synid_vcc_lo>`.

.. _amdgpu_synid_execz:

execz
-----

A single bit flag indicating that the :ref:`exec<amdgpu_synid_exec>` is all zeros.

Note. When GFX10 operates in *wave32* mode, this register reflects state of :ref:`exec_lo<amdgpu_synid_exec>`.

.. _amdgpu_synid_scc:

scc
---

A single bit flag indicating the result of a scalar compare operation.

.. _amdgpu_synid_lds_direct:

lds_direct
----------

A special operand which supplies a 32-bit value
fetched from *LDS* memory using :ref:`m0<amdgpu_synid_m0>` as an address.

.. _amdgpu_synid_null:

null
----

This is a special operand which may be used as a source or a destination.

When used as a destination, the result of the operation is discarded.

When used as a source, it supplies zero value.

GFX10 only.

.. WARNING:: Due to a H/W bug, this operand cannot be used with VALU instructions in first generation of GFX10.

.. _amdgpu_synid_constant:

constant
--------

A set of integer and floating-point *inline* constants and values:

* :ref:`iconst<amdgpu_synid_iconst>`
* :ref:`fconst<amdgpu_synid_fconst>`
* :ref:`ival<amdgpu_synid_ival>`

In contrast with :ref:`literals<amdgpu_synid_literal>`, these operands are encoded as a part of instruction.

If a number may be encoded as either
a :ref:`literal<amdgpu_synid_literal>` or 
a :ref:`constant<amdgpu_synid_constant>`,
assembler selects the latter encoding as more efficient.

.. _amdgpu_synid_iconst:

iconst
~~~~~~

An :ref:`integer number<amdgpu_synid_integer_number>`
encoded as an *inline constant*.

Only a small fraction of integer numbers may be encoded as *inline constants*.
They are enumerated in the table below.
Other integer numbers have to be encoded as :ref:`literals<amdgpu_synid_literal>`.

Integer *inline constants* are converted to
:ref:`expected operand type<amdgpu_syn_instruction_type>`
as described :ref:`here<amdgpu_synid_int_const_conv>`.

    ================================== ====================================
    Value                              Note
    ================================== ====================================
    {0..64}                            Positive integer inline constants.
    {-16..-1}                          Negative integer inline constants.
    ================================== ====================================

.. WARNING:: GFX7 does not support inline constants for *f16* operands.

.. _amdgpu_synid_fconst:

fconst
~~~~~~

A :ref:`floating-point number<amdgpu_synid_floating-point_number>`
encoded as an *inline constant*.

Only a small fraction of floating-point numbers may be encoded as *inline constants*.
They are enumerated in the table below.
Other floating-point numbers have to be encoded as :ref:`literals<amdgpu_synid_literal>`.

Floating-point *inline constants* are converted to
:ref:`expected operand type<amdgpu_syn_instruction_type>`
as described :ref:`here<amdgpu_synid_fp_const_conv>`.

    ===================== ===================================================== ==================
    Value                 Note                                                  Availability
    ===================== ===================================================== ==================
    0.0                   The same as integer constant 0.                       All GPUs
    0.5                   Floating-point constant 0.5                           All GPUs
    1.0                   Floating-point constant 1.0                           All GPUs
    2.0                   Floating-point constant 2.0                           All GPUs
    4.0                   Floating-point constant 4.0                           All GPUs
    -0.5                  Floating-point constant -0.5                          All GPUs
    -1.0                  Floating-point constant -1.0                          All GPUs
    -2.0                  Floating-point constant -2.0                          All GPUs
    -4.0                  Floating-point constant -4.0                          All GPUs
    0.1592                1.0/(2.0*pi). Use only for 16-bit operands.           GFX8, GFX9, GFX10
    0.15915494            1.0/(2.0*pi). Use only for 16- and 32-bit operands.   GFX8, GFX9, GFX10
    0.15915494309189532   1.0/(2.0*pi).                                         GFX8, GFX9, GFX10
    ===================== ===================================================== ==================

.. WARNING:: GFX7 does not support inline constants for *f16* operands.

.. _amdgpu_synid_ival:

ival
~~~~

A symbolic operand encoded as an *inline constant*.
These operands provide read-only access to H/W registers.

    ======================== ================================================ =============
    Syntax                   Note                                             Availability
    ======================== ================================================ =============
    shared_base              Base address of shared memory region.            GFX9, GFX10
    shared_limit             Address of the end of shared memory region.      GFX9, GFX10
    private_base             Base address of private memory region.           GFX9, GFX10
    private_limit            Address of the end of private memory region.     GFX9, GFX10
    pops_exiting_wave_id     A dedicated counter for POPS.                    GFX9, GFX10
    ======================== ================================================ =============

.. _amdgpu_synid_literal:

literal
-------

A literal is a 64-bit value which is encoded as a separate 32-bit dword in the instruction stream.

If a number may be encoded as either
a :ref:`literal<amdgpu_synid_literal>` or 
an :ref:`inline constant<amdgpu_synid_constant>`,
assembler selects the latter encoding as more efficient.

Literals may be specified as :ref:`integer numbers<amdgpu_synid_integer_number>`,
:ref:`floating-point numbers<amdgpu_synid_floating-point_number>` or
:ref:`expressions<amdgpu_synid_expression>`
(expressions are currently supported for 32-bit operands only).

A 64-bit literal value is converted by assembler
to an :ref:`expected operand type<amdgpu_syn_instruction_type>`
as described :ref:`here<amdgpu_synid_lit_conv>`.

An instruction may use only one literal but several operands may refer the same literal.

.. _amdgpu_synid_uimm8:

uimm8
-----

A 8-bit positive :ref:`integer number<amdgpu_synid_integer_number>`.
The value is encoded as part of the opcode so it is free to use.

.. _amdgpu_synid_uimm32:

uimm32
------

A 32-bit positive :ref:`integer number<amdgpu_synid_integer_number>`.
The value is stored as a separate 32-bit dword in the instruction stream.

.. _amdgpu_synid_uimm20:

uimm20
------

A 20-bit positive :ref:`integer number<amdgpu_synid_integer_number>`.

.. _amdgpu_synid_uimm21:

uimm21
------

A 21-bit positive :ref:`integer number<amdgpu_synid_integer_number>`.

.. WARNING:: Assembler currently supports 20-bit offsets only. Use :ref:`uimm20<amdgpu_synid_uimm20>` as a replacement.

.. _amdgpu_synid_simm21:

simm21
------

A 21-bit :ref:`integer number<amdgpu_synid_integer_number>`.

.. WARNING:: Assembler currently supports 20-bit unsigned offsets only. Use :ref:`uimm20<amdgpu_synid_uimm20>` as a replacement.

.. _amdgpu_synid_off:

off
---

A special entity which indicates that the value of this operand is not used.

    ================================== ===================================================
    Syntax                             Description
    ================================== ===================================================
    off                                Indicates an unused operand.
    ================================== ===================================================


.. _amdgpu_synid_number:

Numbers
=======

.. _amdgpu_synid_integer_number:

Integer Numbers
---------------

Integer numbers are 64 bits wide.
They may be specified in binary, octal, hexadecimal and decimal formats:

    ============== ====================================
    Format         Syntax
    ============== ====================================
    Decimal        [-]?[1-9][0-9]*
    Binary         [-]?0b[01]+
    Octal          [-]?0[0-7]+
    Hexadecimal    [-]?0x[0-9a-fA-F]+
    \              [-]?[0x]?[0-9][0-9a-fA-F]*[hH]
    ============== ====================================

Examples:

.. parsed-literal::

  -1234
  0b1010
  010
  0xff
  0ffh

.. _amdgpu_synid_floating-point_number:

Floating-Point Numbers
----------------------

All floating-point numbers are handled as double (64 bits wide).

Floating-point numbers may be specified in hexadecimal and decimal formats:

    ============== ======================================================== ========================================================
    Format         Syntax                                                   Note
    ============== ======================================================== ========================================================
    Decimal        [-]?[0-9]*[.][0-9]*([eE][+-]?[0-9]*)?                    Must include either a decimal separator or an exponent.
    Hexadecimal    [-]0x[0-9a-fA-F]*(.[0-9a-fA-F]*)?[pP][+-]?[0-9a-fA-F]+
    ============== ======================================================== ========================================================

Examples:

.. parsed-literal::

 -1.234
 234e2
 -0x1afp-10
 0x.1afp10

.. _amdgpu_synid_expression:

Expressions
===========

An expression specifies an address or a numeric value.
There are two kinds of expressions:

* :ref:`Absolute<amdgpu_synid_absolute_expression>`.
* :ref:`Relocatable<amdgpu_synid_relocatable_expression>`.

.. _amdgpu_synid_absolute_expression:

Absolute Expressions
--------------------

The value of an absolute expression remains the same after program relocation.
Absolute expressions must not include unassigned and relocatable values
such as labels.

Examples:

.. parsed-literal::

    x = -1
    y = x + 10

.. _amdgpu_synid_relocatable_expression:

Relocatable Expressions
-----------------------

The value of a relocatable expression depends on program relocation.

Note that use of relocatable expressions is limited with branch targets
and 32-bit :ref:`literals<amdgpu_synid_literal>`.

Addition information about relocation may be found :ref:`here<amdgpu-relocation-records>`.

Examples:

.. parsed-literal::

    y = x + 10 // x is not yet defined. Undefined symbols are assumed to be PC-relative.
    z = .

Expression Data Type
--------------------

Expressions and operands of expressions are interpreted as 64-bit integers.

Expressions may include 64-bit :ref:`floating-point numbers<amdgpu_synid_floating-point_number>` (double).
However these operands are also handled as 64-bit integers
using binary representation of specified floating-point numbers.
No conversion from floating-point to integer is performed.

Examples:

.. parsed-literal::

    x = 0.1    // x is assigned an integer 4591870180066957722 which is a binary representation of 0.1.
    y = x + x  // y is a sum of two integer values; it is not equal to 0.2!

Syntax
------

Expressions are composed of
:ref:`symbols<amdgpu_synid_symbol>`,
:ref:`integer numbers<amdgpu_synid_integer_number>`,
:ref:`floating-point numbers<amdgpu_synid_floating-point_number>`,
:ref:`binary operators<amdgpu_synid_expression_bin_op>`,
:ref:`unary operators<amdgpu_synid_expression_un_op>` and subexpressions.

Expressions may also use "." which is a reference to the current PC (program counter).

The syntax of expressions is shown below::

    expr ::= expr binop expr | primaryexpr ;

    primaryexpr ::= '(' expr ')' | symbol | number | '.' | unop primaryexpr ;

    binop ::= '&&'
            | '||'
            | '|'
            | '^'
            | '&'
            | '!'
            | '=='
            | '!='
            | '<>'
            | '<'
            | '<='
            | '>'
            | '>='
            | '<<'
            | '>>'
            | '+'
            | '-'
            | '*'
            | '/'
            | '%' ;

    unop ::= '~'
           | '+'
           | '-'
           | '!' ;

.. _amdgpu_synid_expression_bin_op:

Binary Operators
----------------

Binary operators are described in the following table.
They operate on and produce 64-bit integers.
Operators with higher priority are performed first.

    ========== ========= ===============================================
    Operator   Priority  Meaning
    ========== ========= ===============================================
       \*         5      Integer multiplication.
       /          5      Integer division.
       %          5      Integer signed remainder.
       \+         4      Integer addition.
       \-         4      Integer subtraction.
       <<         3      Integer shift left.
       >>         3      Logical shift right.
       ==         2      Equality comparison.
       !=         2      Inequality comparison.
       <>         2      Inequality comparison.
       <          2      Signed less than comparison.
       <=         2      Signed less than or equal comparison.
       >          2      Signed greater than comparison.
       >=         2      Signed greater than or equal comparison.
      \|          1      Bitwise or.
       ^          1      Bitwise xor.
       &          1      Bitwise and.
       &&         0      Logical and.
       ||         0      Logical or.
    ========== ========= ===============================================

.. _amdgpu_synid_expression_un_op:

Unary Operators
---------------

Unary operators are described in the following table.
They operate on and produce 64-bit integers.

    ========== ===============================================
    Operator   Meaning
    ========== ===============================================
       !       Logical negation.
       ~       Bitwise negation.
       \+      Integer unary plus.
       \-      Integer unary minus.
    ========== ===============================================

.. _amdgpu_synid_symbol:

Symbols
-------

A symbol is a named 64-bit value, representing a relocatable
address or an absolute (non-relocatable) number.

Symbol names have the following syntax:
    ``[a-zA-Z_.][a-zA-Z0-9_$.@]*``

The table below provides several examples of syntax used for symbol definition.

    ================ ==========================================================
    Syntax           Meaning
    ================ ==========================================================
    .globl <S>       Declares a global symbol S without assigning it a value.
    .set <S>, <E>    Assigns the value of an expression E to a symbol S.
    <S> = <E>        Assigns the value of an expression E to a symbol S.
    <S>:             Declares a label S and assigns it the current PC value.
    ================ ==========================================================

A symbol may be used before it is declared or assigned;
unassigned symbols are assumed to be PC-relative.

Addition information about symbols may be found :ref:`here<amdgpu-symbols>`.

.. _amdgpu_synid_conv:

Conversions
===========

This section describes what happens when a 64-bit
:ref:`integer number<amdgpu_synid_integer_number>`, a
:ref:`floating-point numbers<amdgpu_synid_floating-point_number>` or a
:ref:`symbol<amdgpu_synid_symbol>`
is used for an operand which has a different type or size.

Depending on operand kind, this conversion is performed by either assembler or AMDGPU H/W:

* Values encoded as :ref:`inline constants<amdgpu_synid_constant>` are handled by H/W.
* Values encoded as :ref:`literals<amdgpu_synid_literal>` are converted by assembler.

.. _amdgpu_synid_const_conv:

Inline Constants
----------------

.. _amdgpu_synid_int_const_conv:

Integer Inline Constants
~~~~~~~~~~~~~~~~~~~~~~~~

Integer :ref:`inline constants<amdgpu_synid_constant>`
may be thought of as 64-bit
:ref:`integer numbers<amdgpu_synid_integer_number>`;
when used as operands they are truncated to the size of
:ref:`expected operand type<amdgpu_syn_instruction_type>`.
No data type conversions are performed.

Examples:

.. parsed-literal::

    // GFX9

    v_add_u16 v0, -1, 0    // v0 = 0xFFFF
    v_add_f16 v0, -1, 0    // v0 = 0xFFFF (NaN)

    v_add_u32 v0, -1, 0    // v0 = 0xFFFFFFFF
    v_add_f32 v0, -1, 0    // v0 = 0xFFFFFFFF (NaN)

.. _amdgpu_synid_fp_const_conv:

Floating-Point Inline Constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Floating-point :ref:`inline constants<amdgpu_synid_constant>`
may be thought of as 64-bit
:ref:`floating-point numbers<amdgpu_synid_floating-point_number>`;
when used as operands they are converted to a floating-point number of
:ref:`expected operand size<amdgpu_syn_instruction_type>`.

Examples:

.. parsed-literal::

    // GFX9

    v_add_f16 v0, 1.0, 0    // v0 = 0x3C00 (1.0)
    v_add_u16 v0, 1.0, 0    // v0 = 0x3C00

    v_add_f32 v0, 1.0, 0    // v0 = 0x3F800000 (1.0)
    v_add_u32 v0, 1.0, 0    // v0 = 0x3F800000


.. _amdgpu_synid_lit_conv:

Literals
--------

.. _amdgpu_synid_int_lit_conv:

Integer Literals
~~~~~~~~~~~~~~~~

Integer :ref:`literals<amdgpu_synid_literal>`
are specified as 64-bit :ref:`integer numbers<amdgpu_synid_integer_number>`.

When used as operands they are converted to
:ref:`expected operand type<amdgpu_syn_instruction_type>` as described below.

    ============== ============== =============== ====================================================================
    Expected type  Condition      Result          Note
    ============== ============== =============== ====================================================================
    i16, u16, b16  cond(num,16)   num.u16         Truncate to 16 bits.
    i32, u32, b32  cond(num,32)   num.u32         Truncate to 32 bits.
    i64            cond(num,32)   {-1,num.i32}    Truncate to 32 bits and then sign-extend the result to 64 bits.
    u64, b64       cond(num,32)   { 0,num.u32}    Truncate to 32 bits and then zero-extend the result to 64 bits.
    f16            cond(num,16)   num.u16         Use low 16 bits as an f16 value.
    f32            cond(num,32)   num.u32         Use low 32 bits as an f32 value.
    f64            cond(num,32)   {num.u32,0}     Use low 32 bits of the number as high 32 bits
                                                  of the result; low 32 bits of the result are zeroed.
    ============== ============== =============== ====================================================================

The condition *cond(X,S)* indicates if a 64-bit number *X*
can be converted to a smaller size *S* by truncation of upper bits.
There are two cases when the conversion is possible:

* The truncated bits are all 0.
* The truncated bits are all 1 and the value after truncation has its MSB bit set.

Examples of valid literals:

.. parsed-literal::

    // GFX9
                                             // Literal value after conversion:
    v_add_u16 v0, 0xff00, v0                 //   0xff00
    v_add_u16 v0, 0xffffffffffffff00, v0     //   0xff00
    v_add_u16 v0, -256, v0                   //   0xff00
                                             // Literal value after conversion:
    s_bfe_i64 s[0:1], 0xffefffff, s3         //   0xffffffffffefffff
    s_bfe_u64 s[0:1], 0xffefffff, s3         //   0x00000000ffefffff
    v_ceil_f64_e32 v[0:1], 0xffefffff        //   0xffefffff00000000 (-1.7976922776554302e308)

Examples of invalid literals:

.. parsed-literal::

    // GFX9

    v_add_u16 v0, 0x1ff00, v0               // truncated bits are not all 0 or 1
    v_add_u16 v0, 0xffffffffffff00ff, v0    // truncated bits do not match MSB of the result

.. _amdgpu_synid_fp_lit_conv:

Floating-Point Literals
~~~~~~~~~~~~~~~~~~~~~~~

Floating-point :ref:`literals<amdgpu_synid_literal>` are specified as 64-bit
:ref:`floating-point numbers<amdgpu_synid_floating-point_number>`.

When used as operands they are converted to
:ref:`expected operand type<amdgpu_syn_instruction_type>` as described below.

    ============== ============== ================= =================================================================
    Expected type  Condition      Result            Note
    ============== ============== ================= =================================================================
    i16, u16, b16  cond(num,16)   f16(num)          Convert to f16 and use bits of the result as an integer value.
    i32, u32, b32  cond(num,32)   f32(num)          Convert to f32 and use bits of the result as an integer value.
    i64, u64, b64  false          \-                Conversion disabled because of an unclear semantics.
    f16            cond(num,16)   f16(num)          Convert to f16.
    f32            cond(num,32)   f32(num)          Convert to f32.
    f64            true           {num.u32.hi,0}    Use high 32 bits of the number as high 32 bits of the result;
                                                    zero-fill low 32 bits of the result.

                                                    Note that the result may differ from the original number.
    ============== ============== ================= =================================================================

The condition *cond(X,S)* indicates if an f64 number *X* can be converted
to a smaller *S*-bit floating-point type without overflow or underflow.
Precision lost is allowed.

Examples of valid literals:

.. parsed-literal::

    // GFX9

    v_add_f16 v1, 65500.0, v2
    v_add_f32 v1, 65600.0, v2

    // Literal value before conversion: 1.7976931348623157e308 (0x7fefffffffffffff)
    // Literal value after conversion:  1.7976922776554302e308 (0x7fefffff00000000)
    v_ceil_f64 v[0:1], 1.7976931348623157e308

Examples of invalid literals:

.. parsed-literal::

    // GFX9

    v_add_f16 v1, 65600.0, v2    // overflow

.. _amdgpu_synid_exp_conv:

Expressions
~~~~~~~~~~~

Expressions operate with and result in 64-bit integers.

When used as operands they are truncated to
:ref:`expected operand size<amdgpu_syn_instruction_type>`.
No data type conversions are performed.

Examples:

.. parsed-literal::

    // GFX9

    x = 0.1
    v_sqrt_f32 v0, x           // v0 = [low 32 bits of 0.1 (double)]
    v_sqrt_f32 v0, (0.1 + 0)   // the same as above
    v_sqrt_f32 v0, 0.1         // v0 = [0.1 (double) converted to float]

