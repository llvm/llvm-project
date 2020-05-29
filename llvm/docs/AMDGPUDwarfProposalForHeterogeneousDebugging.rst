.. _amdgpu-dwarf-proposal-for-heterogeneous-debugging:

******************************************
DWARF Proposal For Heterogeneous Debugging
******************************************

.. contents::
   :local:

.. warning::

   This document describes a **provisional proposal** for DWARF Version 6
   [:ref:`DWARF <amdgpu-dwarf-DWARF>`] to support heterogeneous debugging. It is
   not currently fully implemented and is subject to change.

.. _amdgpu-dwarf-introduction:

Introduction
============

AMD [:ref:`AMD <amdgpu-dwarf-AMD>`] has been working on supporting heterogeneous
computing through the AMD Radeon Open Compute Platform (ROCm) [:ref:`AMD-ROCm
<amdgpu-dwarf-AMD-ROCm>`]. A heterogeneous computing program can be written in a
high level language such as C++ or Fortran with OpenMP pragmas, OpenCL, or HIP
(a portable C++ programming environment for heterogeneous computing [:ref:`HIP
<amdgpu-dwarf-HIP>`]). A heterogeneous compiler and runtime allows a program to
execute on multiple devices within the same native process. Devices could
include CPUs, GPUs, DSPs, FPGAs, or other special purpose accelerators.
Currently HIP programs execute on systems with CPUs and GPUs.

ROCm is fully open sourced and includes contributions to open source projects
such as LLVM for compilation [:ref:`LLVM <amdgpu-dwarf-LLVM>`] and GDB for
debugging [:ref:`GDB <amdgpu-dwarf-GDB>`], as well as collaboration with other
third party projects such as the GCC compiler [:ref:`GCC <amdgpu-dwarf-GCC>`]
and the Perforce TotalView HPC debugger [:ref:`Perforce-TotalView
<amdgpu-dwarf-Perforce-TotalView>`].

To support debugging heterogeneous programs several features that are not
provided by current DWARF Version 5 [:ref:`DWARF <amdgpu-dwarf-DWARF>`] have
been identified. This document contains a collection of proposals to address
providing those features.

The :ref:`amdgpu-dwarf-motivation` section describes the issues that are being
addressed for heterogeneous computing. That is followed by the
:ref:`amdgpu-dwarf-proposed-changes-relative-to-dwarf-version-5` section
containing the proposed textual changes relative to the DWARF Version 5
standard. Then there is an :ref:`amdgpu-dwarf-examples` section that links to
the AMD GPU specific usage of the features in the proposal that includes an
example. Finally, there is a :ref:`amdgpu-dwarf-references` section. There are a
number of notes included that raise open questions, or provide alternative
approaches considered. The draft proposal seeks to be general in nature and
backwards compatible with DWARF Version 5. Its goal is to be applicable to
meeting the needs of any heterogeneous system and not be vendor or architecture
specific.

A fundamental aspect of the draft proposal is that it allows DWARF expression
location descriptions as stack elements. The draft proposal is based on DWARF
Version 5 and maintains compatibility with DWARF Version 5. After attempting
several alternatives, the current thinking is that such an addition to DWARF
Version 5 is the simplest and cleanest way to support debugging optimized GPU
code. It also appears to be generally useful and may be able to address other
reported DWARF issues, as well as being helpful in providing better optimization
support for non-GPU code.

General feedback on this draft proposal is sought, together with suggestions on
how to clarify, simplify, or organize it before submitting it as a formal DWARF
proposal. The current draft proposal is large and may need to be split into
separate proposals before formal submission. Any suggestions on how best to do
that are appreciated. However, at the initial review stage it is believed there
is value in presenting a unified proposal as there are mutual dependencies
between the various parts that would not be as apparent if it was broken up into
separate independent proposals.

We are in the process of modifying LLVM and GDB to support this draft proposal
which is providing experience and insights. We plan to upstream the changes to
those projects for any final form of the proposal.

The author very much appreciates the input provided so far by many others which
has been incorporated into this current version.

.. _amdgpu-dwarf-motivation:

Motivation
==========

This document proposes a set of backwards compatible extensions to DWARF Version
5 [:ref:`DWARF <amdgpu-dwarf-DWARF>`] for consideration of inclusion into a
future DWARF Version 6 standard to support heterogeneous debugging.

The remainder of this section provides motivation for each proposed feature in
terms of heterogeneous debugging on commercially available AMD GPU hardware
(AMDGPU). The goal is to add support to the AMD [:ref:`AMD <amdgpu-dwarf-AMD>`]
open source Radeon Open Compute Platform (ROCm) [:ref:`AMD-ROCm
<amdgpu-dwarf-AMD-ROCm>`] which is an implementation of the industry standard
for heterogeneous computing devices defined by the Heterogeneous System
Architecture (HSA) Foundation [:ref:`HSA <amdgpu-dwarf-HSA>`]. ROCm includes the
LLVM compiler [:ref:`LLVM <amdgpu-dwarf-LLVM>`] with upstreamed support for
AMDGPU [:ref:`AMDGPU-LLVM <amdgpu-dwarf-AMDGPU-LLVM>`]. The goal is to also add
the GDB debugger [:ref:`GDB <amdgpu-dwarf-GDB>`] with upstreamed support for
AMDGPU [:ref:`AMD-ROCgdb <amdgpu-dwarf-AMD-ROCgdb>`]. In addition, the goal is
to work with third parties to enable support for AMDGPU debugging in the GCC
compiler [:ref:`GCC <amdgpu-dwarf-GCC>`] and the Perforce TotalView HPC debugger
[:ref:`Perforce-TotalView <amdgpu-dwarf-Perforce-TotalView>`].

However, the proposal is intended to be vendor and architecture neutral. It is
believed to apply to other heterogeous hardware devices including GPUs, DSPs,
FPGAs, and other specialized hardware. These collectively include similar
characteristics and requirements as AMDGPU devices. Parts of the proposal can
also apply to traditional CPU hardware that supports large vector registers.
Compilers can map source languages and extensions that describe large scale
parallel execution onto the lanes of the vector registers. This is common in
programming languages used in ML and HPC. The proposal also includes improved
support for optimized code on any architecture. Some of the generalizations may
also benefit other issues that have been raised.

The proposal has evolved though collaboration with many individuals and active
prototyping within the GDB debugger and LLVM compiler. Input has also been very
much appreciated from the developers working on the Perforce TotalView HPC
Debugger and GCC compiler.

The AMDGPU has several features that require additional DWARF functionality in
order to support optimized code.

AMDGPU optimized code may spill vector registers to non-global address space
memory, and this spilling may be done only for lanes that are active on entry
to the subprogram. To support this, a location description that can be created
as a masked select is required. See ``DW_OP_LLVM_select_bit_piece``.

Since the active lane mask may be held in a register, a way to get the value
of a register on entry to a subprogram is required. To support this an
operation that returns the caller value of a register as specified by the Call
Frame Information (CFI) is required. See ``DW_OP_LLVM_call_frame_entry_reg``
and :ref:`amdgpu-dwarf-call-frame-information`.

Current DWARF uses an empty expression to indicate an undefined location
description. Since the masked select composite location description operation
takes more than one location description, it is necessary to have an explicit
way to specify an undefined location description. Otherwise it is not possible
to specify that a particular one of the input location descriptions is
undefined. See ``DW_OP_LLVM_undefined``.

CFI describes restoring callee saved registers that are spilled. Currently CFI
only allows a location description that is a register, memory address, or
implicit location description. AMDGPU optimized code may spill scalar
registers into portions of vector registers. This requires extending CFI to
allow any location description. See
:ref:`amdgpu-dwarf-call-frame-information`.

The vector registers of the AMDGPU are represented as their full wavefront
size, meaning the wavefront size times the dword size. This reflects the
actual hardware and allows the compiler to generate DWARF for languages that
map a thread to the complete wavefront. It also allows more efficient DWARF to
be generated to describe the CFI as only a single expression is required for
the whole vector register, rather than a separate expression for each lane's
dword of the vector register. It also allows the compiler to produce DWARF
that indexes the vector register if it spills scalar registers into portions
of a vector registers.

Since DWARF stack value entries have a base type and AMDGPU registers are a
vector of dwords, the ability to specify that a base type is a vector is
required. See ``DW_AT_LLVM_vector_size``.

If the source language is mapped onto the AMDGPU wavefronts in a SIMT manner,
then the variable DWARF location expressions must compute the location for a
single lane of the wavefront. Therefore, a DWARF operation is required to
denote the current lane, much like ``DW_OP_push_object_address`` denotes the
current object. The ``DW_OP_*piece`` operations only allow literal indices.
Therefore, a way to use a computed offset of an arbitrary location description
(such as a vector register) is required. See ``DW_OP_LLVM_push_lane``,
``DW_OP_LLVM_offset``, ``DW_OP_LLVM_offset_uconst``, and
``DW_OP_LLVM_bit_offset``.

If the source language is mapped onto the AMDGPU wavefronts in a SIMT manner
the compiler can use the AMDGPU execution mask register to control which lanes
are active. To describe the conceptual location of non-active lanes a DWARF
expression is needed that can compute a per lane PC. For efficiency, this is
done for the wavefront as a whole. This expression benefits by having a masked
select composite location description operation. This requires an attribute
for source location of each lane. The AMDGPU may update the execution mask for
whole wavefront operations and so needs an attribute that computes the current
active lane mask. See ``DW_OP_LLVM_select_bit_piece``, ``DW_OP_LLVM_extend``,
``DW_AT_LLVM_lane_pc``, and ``DW_AT_LLVM_active_lane``.

AMDGPU needs to be able to describe addresses that are in different kinds of
memory. Optimized code may need to describe a variable that resides in pieces
that are in different kinds of storage which may include parts of registers,
memory that is in a mixture of memory kinds, implicit values, or be undefined.
DWARF has the concept of segment addresses. However, the segment cannot be
specified within a DWARF expression, which is only able to specify the offset
portion of a segment address. The segment index is only provided by the entity
that specifies the DWARF expression. Therefore, the segment index is a
property that can only be put on complete objects, such as a variable. That
makes it only suitable for describing an entity (such as variable or
subprogram code) that is in a single kind of memory. Therefore, AMDGPU uses
the DWARF concept of address spaces. For example, a variable may be allocated
in a register that is partially spilled to the call stack which is in the
private address space, and partially spilled to the local address space.

DWARF uses the concept of an address in many expression operations but does not
define how it relates to address spaces. For example,
``DW_OP_push_object_address`` pushes the address of an object. Other contexts
implicitly push an address on the stack before evaluating an expression. For
example, the ``DW_AT_use_location`` attribute of the
``DW_TAG_ptr_to_member_type``. The expression that uses the address needs to
do so in a general way and not need to be dependent on the address space of
the address. For example, a pointer to member value may want to be applied to
an object that may reside in any address space.

The number of registers and the cost of memory operations is much higher for
AMDGPU than a typical CPU. The compiler attempts to optimize whole variables
and arrays into registers. Currently DWARF only allows
``DW_OP_push_object_address`` and related operations to work with a global
memory location. To support AMDGPU optimized code it is required to generalize
DWARF to allow any location description to be used. This allows registers, or
composite location descriptions that may be a mixture of memory, registers, or
even implicit values.

DWARF Version 5 does not allow location descriptions to be entries on the
DWARF stack. They can only be the final result of the evaluation of a DWARF
expression. However, by allowing a location description to be a first-class
entry on the DWARF stack it becomes possible to compose expressions containing
both values and location descriptions naturally. It allows objects to be
located in any kind of memory address space, in registers, be implicit values,
be undefined, or a composite of any of these. By extending DWARF carefully,
all existing DWARF expressions can retain their current semantic meaning.
DWARF has implicit conversions that convert from a value that represents an
address in the default address space to a memory location description. This
can be extended to allow a default address space memory location description
to be implicitly converted back to its address value. This allows all DWARF
Version 5 expressions to retain their same meaning, while adding the ability
to explicitly create memory location descriptions in non-default address
spaces and generalizing the power of composite location descriptions to any
kind of location description. See :ref:`amdgpu-dwarf-operation-expressions`.

To allow composition of composite location descriptions, an explicit operation
that indicates the end of the definition of a composite location description
is required. This can be implied if the end of a DWARF expression is reached,
allowing current DWARF expressions to remain legal. See
``DW_OP_LLVM_piece_end``.

The ``DW_OP_plus`` and ``DW_OP_minus`` can be defined to operate on a memory
location description in the default target architecture specific address space
and a generic type value to produce an updated memory location description. This
allows them to continue to be used to offset an address. To generalize
offsetting to any location description, including location descriptions that
describe when bytes are in registers, are implicit, or a composite of these, the
``DW_OP_LLVM_offset``, ``DW_OP_LLVM_offset_uconst``, and
``DW_OP_LLVM_bit_offset`` offset operations are added. Unlike ``DW_OP_plus``,
``DW_OP_plus_uconst``, and ``DW_OP_minus`` arithmetic operations, these do not
define that integer overflow causes wrap-around. The offset operations can
operate on location storage of any size. For example, implicit location storage
could be any number of bits in size. It is simpler to define offsets that exceed
the size of the location storage as being ill-formed, than having to force an
implementation to support potentially infinite precision offsets to allow it to
correctly track a series of positive and negative offsets that may transiently
overflow or underflow, but end up in range. This is simple for the arithmetic
operations as they are defined in terms of two's compliment arithmetic on a base
type of a fixed size.

Having the offset operations allows ``DW_OP_push_object_address`` to push a
location description that may be in a register, or be an implicit value, and the
DWARF expression of ``DW_TAG_ptr_to_member_type`` can contain them to offset
within it. ``DW_OP_LLVM_bit_offset`` generalizes DWARF to work with bit fields
which is not possible in DWARF Version 5.

The DWARF ``DW_OP_xderef*`` operations allow a value to be converted into an
address of a specified address space which is then read. But it provides no
way to create a memory location description for an address in the non-default
address space. For example, AMDGPU variables can be allocated in the local
address space at a fixed address. It is required to have an operation to
create an address in a specific address space that can be used to define the
location description of the variable. Defining this operation to produce a
location description allows the size of addresses in an address space to be
larger than the generic type. See ``DW_OP_LLVM_form_aspace_address``.

If the ``DW_OP_LLVM_form_aspace_address`` operation had to produce a value
that can be implicitly converted to a memory location description, then it
would be limited to the size of the generic type which matches the size of the
default address space. Its value would be unspecified and likely not match any
value in the actual program. By making the result a location description, it
allows a consumer great freedom in how it implements it. The implicit
conversion back to a value can be limited only to the default address space to
maintain compatibility with DWARF Version 5. For other address spaces the
producer can use the new operations that explicitly specify the address space.

``DW_OP_breg*`` treats the register as containing an address in the default
address space. It is required to be able to specify the address space of the
register value. See ``DW_OP_LLVM_aspace_bregx``.

Similarly, ``DW_OP_implicit_pointer`` treats its implicit pointer value as
being in the default address space. It is required to be able to specify the
address space of the pointer value. See
``DW_OP_LLVM_aspace_implicit_pointer``.

Almost all uses of addresses in DWARF are limited to defining location
descriptions, or to be dereferenced to read memory. The exception is
``DW_CFA_val_offset`` which uses the address to set the value of a register.
By defining the CFA DWARF expression as being a memory location description,
it can maintain what address space it is, and that can be used to convert the
offset address back to an address in that address space. See
:ref:`amdgpu-dwarf-call-frame-information`.

This approach allows all existing DWARF to have the identical semantics. It
allows the compiler to explicitly specify the address space it is using. For
example, a compiler could choose to access private memory in a swizzled manner
when mapping a source language to a wavefront in a SIMT manner, or to access
it in an unswizzled manner if mapping the same language with the wavefront
being the thread. It also allows the compiler to mix the address space it uses
to access private memory. For example, for SIMT it can still spill entire
vector registers in an unswizzled manner, while using a swizzled private
memory for SIMT variable access. This approach allows memory location
descriptions for different address spaces to be combined using the regular
``DW_OP_*piece`` operations.

Location descriptions are an abstraction of storage, they give freedom to the
consumer on how to implement them. They allow the address space to encode lane
information so they can be used to read memory with only the memory
description and no extra arguments. The same set of operations can operate on
locations independent of their kind of storage. The ``DW_OP_deref*`` therefore
can be used on any storage kind. ``DW_OP_xderef*`` is unnecessary except to
become a more compact way to convert a non-default address space address
followed by dereferencing it.

In DWARF Version 5 a location description is defined as a single location
description or a location list. A location list is defined as either
effectively an undefined location description or as one or more single
location descriptions to describe an object with multiple places. The
``DW_OP_push_object_address`` and ``DW_OP_call*`` operations can put a
location description on the stack. Furthermore, debugger information entry
attributes such as ``DW_AT_data_member_location``, ``DW_AT_use_location``, and
``DW_AT_vtable_elem_location`` are defined as pushing a location description
on the expression stack before evaluating the expression. However, DWARF
Version 5 only allows the stack to contain values and so only a single memory
address can be on the stack which makes these incapable of handling location
descriptions with multiple places, or places other than memory. Since this
proposal allows the stack to contain location descriptions, the operations are
generalized to support location descriptions that can have multiple places.
This is backwards compatible with DWARF Version 5 and allows objects with
multiple places to be supported. For example, the expression that describes
how to access the field of an object can be evaluated with a location
description that has multiple places and will result in a location description
with multiple places as expected. With this change, the separate DWARF Version
5 sections that described DWARF expressions and location lists have been
unified into a single section that describes DWARF expressions in general.
This unification seems to be a natural consequence and a necessity of allowing
location descriptions to be part of the evaluation stack.

For those familiar with the definition of location descriptions in DWARF
Version 5, the definition in this proposal is presented differently, but does
in fact define the same concept with the same fundamental semantics. However,
it does so in a way that allows the concept to extend to support address
spaces, bit addressing, the ability for composite location descriptions to be
composed of any kind of location description, and the ability to support
objects located at multiple places. Collectively these changes expand the set
of processors that can be supported and improves support for optimized code.

Several approaches were considered, and the one proposed appears to be the
cleanest and offers the greatest improvement of DWARF's ability to support
optimized code. Examining the GDB debugger and LLVM compiler, it appears only
to require modest changes as they both already have to support general use of
location descriptions. It is anticipated that will also be the case for other
debuggers and compilers.

As an experiment, GDB was modified to evaluate DWARF Version 5 expressions
with location descriptions as stack entries and implicit conversions. All GDB
tests have passed, except one that turned out to be an invalid test by DWARF
Version 5 rules. The code in GDB actually became simpler as all evaluation was
on the stack and there was no longer a need to maintain a separate structure
for the location description result. This gives confidence of the backwards
compatibility.

Since the AMDGPU supports languages such as OpenCL [:ref:`OpenCL
<amdgpu-dwarf-OpenCL>`], there is a need to define source language address
classes so they can be used in a consistent way by consumers. It would also be
desirable to add support for using them in defining language types rather than
the current target architecture specific address spaces. See
:ref:`amdgpu-dwarf-segment_addresses`.

A ``DW_AT_LLVM_augmentation`` attribute is added to a compilation unit
debugger information entry to indicate that there is additional target
architecture specific information in the debugging information entries of that
compilation unit. This allows a consumer to know what extensions are present
in the debugger information entries as is possible with the augmentation
string of other sections. The format that should be used for the augmentation
string in the lookup by name table and CFI Common Information Entry is also
recommended to allow a consumer to parse the string when it contains
information from multiple vendors.

The AMDGPU supports programming languages that include online compilation
where the source text may be created at runtime. Therefore, a way to embed the
source text in the debug information is required. For example, the OpenCL
language runtime supports online compilation. See
:ref:`amdgpu-dwarf-line-number-information`.

Support to allow MD5 checksums to be optionally present in the line table is
added. This allows linking together compilation units where some have MD5
checksums and some do not. In DWARF Version 5 the file timestamp and file size
can be optional, but if the MD5 checksum is present it must be valid for all
files. See :ref:`amdgpu-dwarf-line-number-information`.

Support is added for the HIP programming language [:ref:`HIP
<amdgpu-dwarf-HIP>`] which is supported by the AMDGPU. See
:ref:`amdgpu-dwarf-language-names`.

The following sections provide the definitions for the additional operations,
as well as clarifying how existing expression operations, CFI operations, and
attributes behave with respect to generalized location descriptions that
support address spaces and location descriptions that support multiple places.
It has been defined such that it is backwards compatible with DWARF Version 5.
The definitions are intended to fully define well-formed DWARF in a consistent
style based on the DWARF Version 5 specification. Non-normative text is shown
in *italics*.

The names for the new operations, attributes, and constants include "\
``LLVM``\ " and are encoded with vendor specific codes so this proposal can be
implemented as an LLVM vendor extension to DWARF Version 5. If accepted these
names would not include the "\ ``LLVM``\ " and would not use encodings in the
vendor range.

The proposal is described in
:ref:`amdgpu-dwarf-proposed-changes-relative-to-dwarf-version-5` and is
organized to follow the section ordering of DWARF Version 5. It includes notes
to indicate the corresponding DWARF Version 5 sections to which they pertain.
Other notes describe additional changes that may be worth considering, and to
raise questions.

.. _amdgpu-dwarf-proposed-changes-relative-to-dwarf-version-5:

Proposed Changes Relative to DWARF Version 5
============================================

General Description
-------------------

Attribute Types
~~~~~~~~~~~~~~~

.. note::

  This augments DWARF Version 5 section 2.2 and Table 2.2.

The following table provides the additional attributes. See
:ref:`amdgpu-dwarf-debugging-information-entry-attributes`.

.. table:: Attribute names
   :name: amdgpu-dwarf-attribute-names-table

   =========================== ====================================
   Attribute                   Usage
   =========================== ====================================
   ``DW_AT_LLVM_active_lane``  SIMD or SIMT active lanes
   ``DW_AT_LLVM_augmentation`` Compilation unit augmentation string
   ``DW_AT_LLVM_lane_pc``      SIMD or SIMT lane program location
   ``DW_AT_LLVM_lanes``        SIMD or SIMT thread lane count
   ``DW_AT_LLVM_vector_size``  Base type vector size
   =========================== ====================================

.. _amdgpu-dwarf-expressions:

DWARF Expressions
~~~~~~~~~~~~~~~~~

.. note::

  This section, and its nested sections, replaces DWARF Version 5 section 2.5 and
  section 2.6. The new proposed DWARF expression operations are defined as well
  as clarifying the extensions to already existing DWARF Version 5 operations. It is
  based on the text of the existing DWARF Version 5 standard.

DWARF expressions describe how to compute a value or specify a location.

*The evaluation of a DWARF expression can provide the location of an object, the
value of an array bound, the length of a dynamic string, the desired value
itself, and so on.*

The evaluation of a DWARF expression can either result in a value or a location
description:

*value*

  A value has a type and a literal value. It can represent a literal value of
  any supported base type of the target architecture. The base type specifies
  the size and encoding of the literal value.

  .. note::

    It may be desirable to add an implicit pointer base type encoding. It would
    be used for the type of the value that is produced when the ``DW_OP_deref*``
    operation retrieves the full contents of an implicit pointer location
    storage created by the ``DW_OP_implicit_pointer`` or
    ``DW_OP_LLVM_aspace_implicit_pointer`` operations. The literal value would
    record the debugging information entry and byte dispacement specified by the
    associated ``DW_OP_implicit_pointer`` or
    ``DW_OP_LLVM_aspace_implicit_pointer`` operations.

  Instead of a base type, a value can have a distinguished generic type, which
  is an integral type that has the size of an address in the target architecture
  default address space and unspecified signedness.

  *The generic type is the same as the unspecified type used for stack
  operations defined in DWARF Version 4 and before.*

  An integral type is a base type that has an encoding of ``DW_ATE_signed``,
  ``DW_ATE_signed_char``, ``DW_ATE_unsigned``, ``DW_ATE_unsigned_char``,
  ``DW_ATE_boolean``, or any target architecture defined integral encoding in
  the inclusive range ``DW_ATE_lo_user`` to ``DW_ATE_hi_user``.

  .. note::

    It is unclear if ``DW_ATE_address`` is an integral type. GDB does not seem
    to consider it as integral.

*location description*

  *Debugging information must provide consumers a way to find the location of
  program variables, determine the bounds of dynamic arrays and strings, and
  possibly to find the base address of a subprogram’s stack frame or the return
  address of a subprogram. Furthermore, to meet the needs of recent computer
  architectures and optimization techniques, debugging information must be able
  to describe the location of an object whose location changes over the object’s
  lifetime, and may reside at multiple locations simultaneously during parts of
  an object's lifetime.*

  Information about the location of program objects is provided by location
  descriptions.

  Location descriptions can consist of one or more single location descriptions.

  A single location description specifies the location storage that holds a
  program object and a position within the location storage where the program
  object starts. The position within the location storage is expressed as a bit
  offset relative to the start of the location storage.

  A location storage is a linear stream of bits that can hold values. Each
  location storage has a size in bits and can be accessed using a zero-based bit
  offset. The ordering of bits within a location storage uses the bit numbering
  and direction conventions that are appropriate to the current language on the
  target architecture.

  There are five kinds of location storage:

  *memory location storage*
    Corresponds to the target architecture memory address spaces.

  *register location storage*
    Corresponds to the target architecture registers.

  *implicit location storage*
    Corresponds to fixed values that can only be read.

  *undefined location storage*
    Indicates no value is available and therefore cannot be read or written.

  *composite location storage*
    Allows a mixture of these where some bits come from one location storage and
    some from another location storage, or from disjoint parts of the same
    location storage.

  .. note::

    It may be better to add an implicit pointer location storage kind used by
    the ``DW_OP_implicit_pointer`` and ``DW_OP_LLVM_aspace_implicit_pointer``
    operations. It would specify the debugger information entry and byte offset
    provided by the operations.

  *Location descriptions are a language independent representation of addressing
  rules. They are created using DWARF operation expressions of arbitrary
  complexity. They can be the result of evaluting a debugger information entry
  attribute that specifies an operation expression. In this usage they can
  describe the location of an object as long as its lifetime is either static or
  the same as the lexical block (see DWARF Version 5 section 3.5) that owns it,
  and it does not move during its lifetime. They can be the result of evaluating
  a debugger information entry attribute that specifies a location list
  expression. In this usage they can describe the location of an object that has
  a limited lifetime, changes its location during its lifetime, or has multiple
  locations over part or all of its lifetime.*

  If a location description has more than one single location description, the
  DWARF expression is ill-formed if the object value held in each single
  location description's position within the associated location storage is not
  the same value, except for the parts of the value that are uninitialized.

  *A location description that has more than one single location description can
  only be created by a location list expression that has overlapping program
  location ranges, or certain expression operations that act on a location
  description that has more than one single location description. There are no
  operation expression operations that can directly create a location
  description with more than one single location description.*

  *A location description with more than one single location description can be
  used to describe objects that reside in more than one piece of storage at the
  same time. An object may have more than one location as a result of
  optimization. For example, a value that is only read may be promoted from
  memory to a register for some region of code, but later code may revert to
  reading the value from memory as the register may be used for other purposes.
  For the code region where the value is in a register, any change to the object
  value must be made in both the register and the memory so both regions of code
  will read the updated value.*

  *A consumer of a location description with more than one single location
  description can read the object's value from any of the single location
  descriptions (since they all refer to location storage that has the same
  value), but must write any changed value to all the single location
  descriptions.*

A DWARF expression can either be encoded as a operation expression (see
:ref:`amdgpu-dwarf-operation-expressions`), or as a location list expression
(see :ref:`amdgpu-dwarf-location-list-expressions`).

A DWARF expression is evaluated in the context of:

*A current subprogram*
  This may be used in the evaluation of register access operations to support
  virtual unwinding of the call stack (see
  :ref:`amdgpu-dwarf-call-frame-information`).

*A current program location*
  This may be used in the evaluation of location list expressions to select
  amongst multiple program location ranges. It should be the program location
  corresponding to the current subprogram. If the current subprogram was reached
  by virtual call stack unwinding, then the program location will correspond to
  the associated call site.

*An initial stack*
  This is a list of values or location descriptions that will be pushed on the
  operation expression evaluation stack in the order provided before evaluation
  of an operation expression starts.

  Some debugger information entries have attributes that evaluate their DWARF
  expression value with initial stack entries. In all other cases the initial
  stack is empty.

When a DWARF expression is evaluated, it may be specified whether a value or
location description is required as the result kind.

If a result kind is specified, and the result of the evaluation does not match
the specified result kind, then the implicit conversions described in
:ref:`amdgpu-dwarf-memory-location-description-operations` are performed if
valid. Otherwise, the DWARF expression is ill-formed.

.. _amdgpu-dwarf-operation-expressions:

DWARF Operation Expressions
+++++++++++++++++++++++++++

An operation expression is comprised of a stream of operations, each consisting
of an opcode followed by zero or more operands. The number of operands is
implied by the opcode.

Operations represent a postfix operation on a simple stack machine. Each stack
entry can hold either a value or a location description. Operations can act on
entries on the stack, including adding entries and removing entries. If the kind
of a stack entry does not match the kind required by the operation and is not
implicitly convertible to the required kind (see
:ref:`amdgpu-dwarf-memory-location-description-operations`), then the DWARF
operation expression is ill-formed.

Evaluation of an operation expression starts with an empty stack on which the
entries from the initial stack provided by the context are pushed in the order
provided. Then the operations are evaluated, starting with the first operation
of the stream, until one past the last operation of the stream is reached. The
result of the evaluation is:

* If evaluation of the DWARF expression requires a location description, then:

  * If the stack is empty, the result is a location description with one
    undefined location description.

    *This rule is for backwards compatibility with DWARF Version 5 which has no
    explicit operation to create an undefined location description, and uses an
    empty operation expression for this purpose.*

  * If the top stack entry is a location description, or can be converted
    to one, then the result is that, possibly converted, location description.
    Any other entries on the stack are discarded.

  * Otherwise the DWARF expression is ill-formed.

    .. note::

      Could define this case as returning an implicit location description as
      if the ``DW_OP_implicit`` operation is performed.

* If evaluation of the DWARF expression requires a value, then:

  * If the top stack entry is a value, or can be converted to one, then the
    result is that, possibly converted, value. Any other entries on the stack
    are discarded.

  * Otherwise the DWARF expression is ill-formed.

* If evaluation of the DWARF expression does not specify if a value or location
  description is required, then:

  * If the stack is empty, the result is a location description with one
    undefined location description.

    *This rule is for backwards compatibility with DWARF Version 5 which has no
    explicit operation to create an undefined location description, and uses an
    empty operation expression for this purpose.*

    .. note::

      This rule is consistent with the rule above for when a location
      description is requested. However, GDB appears to report this as an error
      and no GDB tests appear to cause an empty stack for this case.

  * Otherwise, the top stack entry is returned. Any other entries on the stack
    are discarded.

An operation expression is encoded as a byte block with some form of prefix that
specifies the byte count. It can be used:

* as the value of a debugging information entry attribute that is encoded using
  class ``exprloc`` (see DWARF Version 5 section 7.5.5),

* as the operand to certain operation expression operations,

* as the operand to certain call frame information operations (see
  :ref:`amdgpu-dwarf-call-frame-information`),

* and in location list entries (see
  :ref:`amdgpu-dwarf-location-list-expressions`).

.. _amdgpu-dwarf-stack-operations:

Stack Operations
################

The following operations manipulate the DWARF stack. Operations that index the
stack assume that the top of the stack (most recently added entry) has index 0.
They allow the stack entries to be either a value or location description.

If any stack entry accessed by a stack operation is an incomplete composite
location description (see
:ref:`amdgpu-dwarf-composite-location-description-operations`), then the DWARF
expression is ill-formed.

.. note::

  These operations now support stack entries that are values and location
  descriptions.

.. note::

  If it is desired to also make them work with incomplete composite location
  descriptions, then would need to define that the composite location storage
  specified by the incomplete composite location description is also replicated
  when a copy is pushed. This ensures that each copy of the incomplete composite
  location description can update the composite location storage they specify
  independently.

1.  ``DW_OP_dup``

    ``DW_OP_dup`` duplicates the stack entry at the top of the stack.

2.  ``DW_OP_drop``

    ``DW_OP_drop`` pops the stack entry at the top of the stack and discards it.

3.  ``DW_OP_pick``

    ``DW_OP_pick`` has a single unsigned 1-byte operand that represents an index
    I. A copy of the stack entry with index I is pushed onto the stack.

4.  ``DW_OP_over``

    ``DW_OP_over`` pushes a copy of the entry with index 1.

    *This is equivalent to a ``DW_OP_pick 1`` operation.*

5.  ``DW_OP_swap``

    ``DW_OP_swap`` swaps the top two stack entries. The entry at the top of the
    stack becomes the second stack entry, and the second stack entry becomes the
    top of the stack.

6.  ``DW_OP_rot``

    ``DW_OP_rot`` rotates the first three stack entries. The entry at the top of
    the stack becomes the third stack entry, the second entry becomes the top of
    the stack, and the third entry becomes the second entry.

.. _amdgpu-dwarf-control-flow-operations:

Control Flow Operations
#######################

The following operations provide simple control of the flow of a DWARF operation
expression.

1.  ``DW_OP_nop``

    ``DW_OP_nop`` is a place holder. It has no effect on the DWARF stack
    entries.

2.  ``DW_OP_le``, ``DW_OP_ge``, ``DW_OP_eq``, ``DW_OP_lt``, ``DW_OP_gt``,
    ``DW_OP_ne``

    .. note::

      The same as in DWARF Version 5 section 2.5.1.5.

3.  ``DW_OP_skip``

    ``DW_OP_skip`` is an unconditional branch. Its single operand is a 2-byte
    signed integer constant. The 2-byte constant is the number of bytes of the
    DWARF expression to skip forward or backward from the current operation,
    beginning after the 2-byte constant.

    If the updated position is at one past the end of the last operation, then
    the operation expression evaluation is complete.

    Otherwise, the DWARF expression is ill-formed if the updated operation
    position is not in the range of the first to last operation inclusive, or
    not at the start of an operation.

4.  ``DW_OP_bra``

    ``DW_OP_bra`` is a conditional branch. Its single operand is a 2-byte signed
    integer constant. This operation pops the top of stack. If the value popped
    is not the constant 0, the 2-byte constant operand is the number of bytes of
    the DWARF operation expression to skip forward or backward from the current
    operation, beginning after the 2-byte constant.

    If the updated position is at one past the end of the last operation, then
    the operation expression evaluation is complete.

    Otherwise, the DWARF expression is ill-formed if the updated operation
    position is not in the range of the first to last operation inclusive, or
    not at the start of an operation.

5.  ``DW_OP_call2, DW_OP_call4, DW_OP_call_ref``

    ``DW_OP_call2``, ``DW_OP_call4``, and ``DW_OP_call_ref`` perform DWARF
    procedure calls during evaluation of a DWARF expression.

    ``DW_OP_call2`` and ``DW_OP_call4``, have one operand that is a 2- or 4-byte
    unsigned offset, respectively, of a debugging information entry D in the
    current compilation unit.

    ``DW_OP_call_ref`` has one operand that is a 4-byte unsigned value in the
    32-bit DWARF format, or an 8-byte unsigned value in the 64-bit DWARF format,
    that represents an offset of a debugging information entry D in a
    ``.debug_info`` section, which may be contained in an executable or shared
    object file other than that containing the operation. For references from
    one executable or shared object file to another, the relocation must be
    performed by the consumer.

    .. note:

      It is unclear how crossing from one executable or shared object file to
      another can work. How would a consumer know which executable or shared
      object file is being referenced? In an ELF file the DWARF is in a
      non-ALLOC segment so standard dynamic relocations cannot be used.

    *Operand interpretation of* ``DW_OP_call2``\ *,* ``DW_OP_call4``\ *, and*
    ``DW_OP_call_ref`` *is exactly like that for* ``DW_FORM_ref2``\ *,
    ``DW_FORM_ref4``\ *, and* ``DW_FORM_ref_addr``\ *, respectively.*

    The call operation is evaluated by:

    * If D has a ``DW_AT_location`` attribute that is encoded as a ``exprloc``
      that specifies an operation expression E, then execution of the current
      operation expression continues from the first operation of E. Execution
      continues until one past the last operation of E is reached, at which
      point execution continues with the operation following the call operation.
      Since E is evaluated on the same stack as the call, E can use, add, and/or
      remove entries already on the stack.

      *Values on the stack at the time of the call may be used as parameters by
      the called expression and values left on the stack by the called expression
      may be used as return values by prior agreement between the calling and
      called expressions.*

    * If D has a ``DW_AT_location`` attribute that is encoded as a ``loclist`` or
      ``loclistsptr``, then the specified location list expression E is
      evaluated, and the resulting location description is pushed on the stack.
      The evaluation of E uses a context that has the same current frame and
      current program location as the current operation expression, but an empty
      initial stack.

      .. note::

        This rule avoids having to define how to execute a matched location list
        entry operation expression on the same stack as the call when there are
        multiple matches. But it allows the call to obtain the location
        description for a variable or formal parameter which may use a location
        list expression.

        An alternative is to treat the case when D has a ``DW_AT_location``
        attribute that is encoded as a ``loclist`` or ``loclistsptr``, and the
        specified location list expression E' matches a single location list
        entry with operation expression E, the same as the ``exprloc`` case and
        evaluate on the same stack.

        But this is not attractive as if the attribute is for a variable that
        happens to end with a non-singleton stack, it will not simply put a
        location description on the stack. Presumably the intent of using
        ``DW_OP_call*`` on a variable or formal parameter debugger information
        entry is to push just one location description on the stack. That
        location description may have more than one single location description.

        The previous rule for ``exprloc`` also has the same problem as normally
        a variable or formal parameter location expression may leave multiple
        entries on the stack and only return the top entry.

        GDB implements ``DW_OP_call*`` by always executing E on the same stack.
        If the location list has multiple matching entries, it simply picks the
        first one and ignores the rest. This seems fundementally at odds with
        the desire to supporting multiple places for variables.

        So, it feels like ``DW_OP_call*`` should both support pushing a location
        description on the stack for a variable or formal parameter, and also
        support being able to execute an operation expression on the same stack.
        Being able to specify a different operation expression for different
        program locations seems a desirable feature to retain.

        A solution to that is to have a distinct ``DW_AT_LLVM_proc`` attribute
        for the ``DW_TAG_dwarf_procedure`` debugging information entry. Then the
        ``DW_AT_location`` attribute expression is always executed separately
        and pushes a location description (that may have multiple single
        location descriptions), and the ``DW_AT_LLVM_proc`` attribute expression
        is always executed on the same stack and can leave anything on the
        stack.

        The ``DW_AT_LLVM_proc`` attribute could have the new classes
        ``exprproc``, ``loclistproc``, and ``loclistsptrproc`` to indicate that
        the expression is executed on the same stack. ``exprproc`` is the same
        encoding as ``exprloc``. ``loclistproc`` and ``loclistsptrproc`` are the
        same encoding as their non-\ ``proc`` counterparts except the DWARF is
        ill-formed if the location list does not match exactly one location list
        entry and a default entry is required. These forms indicate explicitly
        that the matched single operation expression must be executed on the
        same stack. This is better than ad hoc special rules for ``loclistproc``
        and ``loclistsptrproc`` which are currently clearly defined to always
        return a location description. The producer then explicitly indicates
        the intent through the attribute classes.

        Such a change would be a breaking change for how GDB implements
        ``DW_OP_call*``. However, are the breaking cases actually occurring in
        practice? GDB could implement the current approach for DWARF Version 5,
        and the new semantics for DWARF Version 6 which has been done for some
        other features.

        Another option is to limit the execution to be on the same stack only to
        the evaluation of an expression E that is the value of a
        ``DW_AT_location`` attribute of a ``DW_TAG_dwarf_procedure`` debugging
        information entry. The DWARF would be ill-formed if E is a location list
        expression that does not match exactly one location list entry. In all
        other cases the evaluation of an expression E that is the value of a
        ``DW_AT_location`` attribute would evaluate E with a context that has
        the same current frame and current program location as the current
        operation expression, but an empty initial stack, and push the resulting
        location description on the stack.

    * If D has a ``DW_AT_const_value`` attribute with a value V, then it is as
      if a ``DW_OP_implicit_value V`` operation was executed.

      *This allows a call operation to be used to compute the location
      description for any variable or formal parameter regardless of whether the
      producer has optimized it to a constant. This is consistent with the
      ``DW_OP_implicit_pointer`` operation.*

      .. note::

        Alternatively, could deprecate using ``DW_AT_const_value`` for
        ``DW_TAG_variable`` and ``DW_TAG_formal_parameter`` debugger information
        entries that are constants and instead use ``DW_AT_location`` with an
        operation expression that results in a location description with one
        implicit location description. Then this rule would not be required.

    * Otherwise, there is no effect and no changes are made to the stack.

      .. note::

        In DWARF Version 5, if D does not have a ``DW_AT_location`` then
        ``DW_OP_call*`` is defined to have no effect. It is unclear that this is
        the right definition as a producer should be able to rely on using
        ``DW_OP_call*`` to get a location description for any non-\
        ``DW_TAG_dwarf_procedure`` debugging information entries. Also, the
        producer should not be creating DWARF with ``DW_OP_call*`` to a
        ``DW_TAG_dwarf_procedure`` that does not have a ``DW_AT_location``
        attribute. So, should this case be defined as an ill-formed DWARF
        expression?

    *The* ``DW_TAG_dwarf_procedure`` *debugging information entry can be used to
    define DWARF procedures that can be called.*

.. _amdgpu-dwarf-value-operations:

Value Operations
################

This section describes the operations that push values on the stack.

Each value stack entry has a type and a literal value and can represent a
literal value of any supported base type of the target architecture. The base
type specifies the size and encoding of the literal value.

Instead of a base type, value stack entries can have a distinguished generic
type, which is an integral type that has the size of an address in the target
architecture default address space and unspecified signedness.

*The generic type is the same as the unspecified type used for stack operations
defined in DWARF Version 4 and before.*

An integral type is a base type that has an encoding of ``DW_ATE_signed``,
``DW_ATE_signed_char``, ``DW_ATE_unsigned``, ``DW_ATE_unsigned_char``,
``DW_ATE_boolean``, or any target architecture defined integral encoding in the
inclusive range ``DW_ATE_lo_user`` to ``DW_ATE_hi_user``.

.. note::

  Unclear if ``DW_ATE_address`` is an integral type. GDB does not seem to
  consider it as integral.

.. _amdgpu-dwarf-literal-operations:

Literal Operations
^^^^^^^^^^^^^^^^^^

The following operations all push a literal value onto the DWARF stack.

Operations other than ``DW_OP_const_type`` push a value V with the generic type.
If V is larger than the generic type, then V is truncated to the generic type
size and the low-order bits used.

1.  ``DW_OP_lit0``, ``DW_OP_lit1``, ..., ``DW_OP_lit31``

    ``DW_OP_lit<N>`` operations encode an unsigned literal value N from 0
    through 31, inclusive. They push the value N with the generic type.

2.  ``DW_OP_const1u``, ``DW_OP_const2u``, ``DW_OP_const4u``, ``DW_OP_const8u``

    ``DW_OP_const<N>u`` operations have a single operand that is a 1, 2, 4, or
    8-byte unsigned integer constant U, respectively. They push the value U with
    the generic type.

3.  ``DW_OP_const1s``, ``DW_OP_const2s``, ``DW_OP_const4s``, ``DW_OP_const8s``

    ``DW_OP_const<N>s`` operations have a single operand that is a 1, 2, 4, or
    8-byte signed integer constant S, respectively. They push the value S with
    the generic type.

4.  ``DW_OP_constu``

    ``DW_OP_constu`` has a single unsigned LEB128 integer operand N. It pushes
    the value N with the generic type.

5.  ``DW_OP_consts``

    ``DW_OP_consts`` has a single signed LEB128 integer operand N. It pushes the
    value N with the generic type.

6.  ``DW_OP_constx``

    ``DW_OP_constx`` has a single unsigned LEB128 integer operand that
    represents a zero-based index into the ``.debug_addr`` section relative to
    the value of the ``DW_AT_addr_base`` attribute of the associated compilation
    unit. The value N in the ``.debug_addr`` section has the size of the generic
    type. It pushes the value N with the generic type.

    *The* ``DW_OP_constx`` *operation is provided for constants that require
    link-time relocation but should not be interpreted by the consumer as a
    relocatable address (for example, offsets to thread-local storage).*

9.  ``DW_OP_const_type``

    ``DW_OP_const_type`` has three operands. The first is an unsigned LEB128
    integer that represents the offset of a debugging information entry D in the
    current compilation unit, that provides the type of the constant value. The
    second is a 1-byte unsigned integral constant S. The third is a block of
    bytes B, with a length equal to S.

    T is the bit size of the type D. The least significant T bits of B are
    interpreted as a value V of the type D. It pushes the value V with the type
    D.

    The DWARF is ill-formed if D is not a ``DW_TAG_base_type`` debugging
    information entry, or if T divided by 8 and rounded up to a multiple of 8
    (the byte size) is not equal to S.

    *While the size of the byte block B can be inferred from the type D
    definition, it is encoded explicitly into the operation so that the
    operation can be parsed easily without reference to the* ``.debug_info``
    *section.*

10. ``DW_OP_LLVM_push_lane`` *New*

    ``DW_OP_LLVM_push_lane`` pushes a value with the generic type that is the
    target architecture specific lane identifier of the thread of execution for
    which a user presented expression is currently being evaluated.

    *For languages that are implemented using a SIMD or SIMT execution model,
    this is the lane number that corresponds to the source language thread of
    execution upon which the user is focused.*

.. _amdgpu-dwarf-arithmetic-logical-operations:

Arithmetic and Logical Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

  This section is the same as DWARF Version 5 section 2.5.1.4.

.. _amdgpu-dwarf-type-conversions-operations:

Type Conversion Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

  This section is the same as DWARF Version 5 section 2.5.1.6.

.. _amdgpu-dwarf-general-operations:

Special Value Operations
^^^^^^^^^^^^^^^^^^^^^^^^

There are these special value operations currently defined:

1.  ``DW_OP_regval_type``

    ``DW_OP_regval_type`` has two operands. The first is an unsigned LEB128
    integer that represents a register number R. The second is an unsigned
    LEB128 integer that represents the offset of a debugging information entry D
    in the current compilation unit, that provides the type of the register
    value.

    The contents of register R are interpreted as a value V of the type D. The
    value V is pushed on the stack with the type D.

    The DWARF is ill-formed if D is not a ``DW_TAG_base_type`` debugging
    information entry, or if the size of type D is not the same as the size of
    register R.

    .. note::

      Should DWARF allow the type D to be a different size to the size of the
      register R? Requiring them to be the same bit size avoids any issue of
      conversion as the bit contents of the register is simply interpreted as a
      value of the specified type. If a conversion is wanted it can be done
      explicitly using a ``DW_OP_convert`` operation.

      GDB has a per register hook that allows a target specific conversion on a
      register by register basis. It defaults to truncation of bigger registers,
      and to actually reading bytes from the next register (or reads out of
      bounds for the last register) for smaller registers. There are no GDB
      tests that read a register out of bounds (except an illegal hand written
      assembly test).

2.  ``DW_OP_deref``

    The ``DW_OP_deref`` operation pops one stack entry that must be a location
    description L.

    A value of the bit size of the generic type is retrieved from the location
    storage specified by L. The value V retrieved is pushed on the stack with
    the generic type.

    If any bit of the value is retrieved from the undefined location storage, or
    the offset of any bit exceeds the size of the location storage specified by
    L, then the DWARF expression is ill-formed.

    See :ref:`amdgpu-dwarf-implicit-location-descriptions` for special rules
    concerning implicit location descriptions created by the
    ``DW_OP_implicit_pointer`` and ``DW_OP_LLVM_implicit_aspace_pointer``
    operations.

    *If L, or the location description of any composite location description
    part that is a subcomponent of L, has more than one single location
    description, then any one of them can be selected as they are required to
    all have the same value. For any single location description SL, bits are
    retrieved from the associated storage location starting at the bit offset
    specified by SL. For a composite location description, the retrieved bits
    are the concatenation of the N bits from each composite location part PL,
    where N is limited to the size of PL.*

3.  ``DW_OP_deref_size``

    ``DW_OP_deref_size`` has a single 1-byte unsigned integral constant that
    represents a byte result size S.

    It pops one stack entry that must be a location description L.

    T is the smaller of the generic type size and S scaled by 8 (the byte size).
    A value V of T bits is retrieved from the location storage specified by L.
    If V is smaller than the size of the generic type, V is zero-extended to the
    generic type size. V is pushed onto the stack with the generic type.

    The DWARF expression is ill-formed if any bit of the value is retrieved from
    the undefined location storage, or if the offset of any bit exceeds the size
    of the location storage specified by L.

    .. note::

      Truncating the value when S is larger than the generic type matches what
      GDB does. This allows the generic type size to not be a integral byte
      size. It does allow S to be arbitrarily large. Should S be restricted to
      the size of the generic type rounded up to a multiple of 8?

    See :ref:`amdgpu-dwarf-implicit-location-descriptions` for special rules
    concerning implicit location descriptions created by the
    ``DW_OP_implicit_pointer`` and ``DW_OP_LLVM_implicit_aspace_pointer``
    operations.

4.  ``DW_OP_deref_type``

    ``DW_OP_deref_type`` has two operands. The first is a 1-byte unsigned
    integral constant S. The second is an unsigned LEB128 integer that
    represents the offset of a debugging information entry D in the current
    compilation unit, that provides the type of the result value.

    It pops one stack entry that must be a location description L. T is the bit
    size of the type D. A value V of T bits is retrieved from the location
    storage specified by L. V is pushed on the stack with the type D.

    The DWARF is ill-formed if D is not a ``DW_TAG_base_type`` debugging
    information entry, if T divided by 8 and rounded up to a multiple of 8 (the
    byte size) is not equal to S, if any bit of the value is retrieved from the
    undefined location storage, or if the offset of any bit exceeds the size of
    the location storage specified by L.

    See :ref:`amdgpu-dwarf-implicit-location-descriptions` for special rules
    concerning implicit location descriptions created by the
    ``DW_OP_implicit_pointer`` and ``DW_OP_LLVM_implicit_aspace_pointer``
    operations.

    *While the size of the pushed value V can be inferred from the type D
    definition, it is encoded explicitly into the operation so that the
    operation can be parsed easily without reference to the* ``.debug_info``
    *section.*

    .. note::

      It is unclear why the operand S is needed. Unlike ``DW_OP_const_type``,
      the size is not needed for parsing. Any evaluation needs to get the base
      type to record with the value to know its encoding and bit size.

      This definition allows the base type to be a bit size since there seems no
      reason to restrict it.

5.  ``DW_OP_xderef`` *Deprecated*

    ``DW_OP_xderef`` pops two stack entries. The first must be an integral type
    value that represents an address A. The second must be an integral type
    value that represents a target architecture specific address space
    identifier AS.

    The operation is equivalent to performing ``DW_OP_swap;
    DW_OP_LLVM_form_aspace_address; DW_OP_deref``. The value V retrieved is left
    on the stack with the generic type.

    *This operation is deprecated as the* ``DW_OP_LLVM_form_aspace_address``
    *operation can be used and provides greater expressiveness.*

6.  ``DW_OP_xderef_size`` *Deprecated*

    ``DW_OP_xderef_size`` has a single 1-byte unsigned integral constant that
    represents a byte result size S.

    It pops two stack entries. The first must be an integral type value that
    represents an address A. The second must be an integral type value that
    represents a target architecture specific address space identifier AS.

    The operation is equivalent to performing ``DW_OP_swap;
    DW_OP_LLVM_form_aspace_address; DW_OP_deref_size S``. The zero-extended
    value V retrieved is left on the stack with the generic type.

    *This operation is deprecated as the* ``DW_OP_LLVM_form_aspace_address``
    *operation can be used and provides greater expressiveness.*

7.  ``DW_OP_xderef_type`` *Deprecated*

    ``DW_OP_xderef_type`` has two operands. The first is a 1-byte unsigned
    integral constant S. The second operand is an unsigned LEB128
    integer R that represents the offset of a debugging information entry D in
    the current compilation unit, that provides the type of the result value.

    It pops two stack entries. The first must be an integral type value that
    represents an address A. The second must be an integral type value that
    represents a target architecture specific address space identifier AS.

    The operation is equivalent to performing ``DW_OP_swap;
    DW_OP_LLVM_form_aspace_address; DW_OP_deref_type S R``. The value V
    retrieved is left on the stack with the type D.

    *This operation is deprecated as the* ``DW_OP_LLVM_form_aspace_address``
    *operation can be used and provides greater expressiveness.*

8.  ``DW_OP_entry_value`` *Deprecated*

    ``DW_OP_entry_value`` pushes the value that the described location held upon
    entering the current subprogram.

    It has two operands. The first is an unsigned LEB128 integer S. The second
    is a block of bytes, with a length equal S, interpreted as a DWARF
    operation expression E.

    E is evaluated as if it had been evaluated upon entering the current
    subprogram with an empty initial stack.

    .. note::

      It is unclear what this means. What is the current program location and
      current frame that must be used? Does this require reverse execution so
      the register and memory state are as it was on entry to the current
      subprogram?

    The DWARF expression is ill-formed if the evaluation of E executes a
    ``DW_OP_push_object_address`` operation.

    If the result of E is a location description with one register location
    description (see :ref:`amdgpu-dwarf-register-location-descriptions`),
    ``DW_OP_entry_value`` pushes the value that register had upon entering the
    current subprogram. The value entry type is the target architecture register
    base type. If the register value is undefined or the register location
    description bit offset is not 0, then the DWARF expression is ill-formed.

    *The register location description provides a more compact form for the case
    where the value was in a register on entry to the subprogram.*

    If the result of E is a value V, ``DW_OP_entry_value`` pushes V on the
    stack.

    Otherwise, the DWARF expression is ill-formed.

    *The values needed to evaluate* ``DW_OP_entry_value`` *could be obtained in
    several ways. The consumer could suspend execution on entry to the
    subprogram, record values needed by* ``DW_OP_entry_value`` *expressions
    within the subprogram, and then continue. When evaluating*
    ``DW_OP_entry_value``\ *, the consumer would use these recorded values
    rather than the current values. Or, when evaluating* ``DW_OP_entry_value``\
    *, the consumer could virtually unwind using the Call Frame Information
    (see* :ref:`amdgpu-dwarf-call-frame-information`\ *) to recover register
    values that might have been clobbered since the subprogram entry point.*

    *The* ``DW_OP_entry_value`` *operation is deprecated as its main usage is
    provided by other means. DWARF Version 5 added the*
    ``DW_TAG_call_site_parameter`` *debugger information entry for call sites
    that has* ``DW_AT_call_value``\ *,* ``DW_AT_call_data_location``\ *, and*
    ``DW_AT_call_data_value`` *attributes that provide DWARF expressions to
    compute actual parameter values at the time of the call, and requires the
    producer to ensure the expressions are valid to evaluate even when virtually
    unwound. The* ``DW_OP_LLVM_call_frame_entry_reg`` *operation provides access
    to registers in the virtually unwound calling frame.*

    .. note::

      It is unclear why this operation is defined this way. How would a consumer
      know what values have to be saved on entry to the subprogram? Does it have
      to parse every expression of every ``DW_OP_entry_value`` operation to
      capture all the possible results needed? Or does it have to implement
      reverse execution so it can evaluate the expression in the context of the
      entry of the subprogram so it can obtain the entry point register and
      memory values? Or does the compiler somehow instruct the consumer how to
      create the saved copies of the variables on entry?

      If the expression is simply using existing variables, then it is just a
      regular expression and no special operation is needed. If the main purpose
      is only to read the entry value of a register using CFI then it would be
      better to have an operation that explicitly does just that such as the
      proposed ``DW_OP_LLVM_call_frame_entry_reg`` operation.

      GDB only seems to implement ``DW_OP_entry_value`` when E is exactly
      ``DW_OP_reg*`` or ``DW_OP_breg*; DW_OP_deref*``. It evaluates E in the
      context of the calling subprogram and the calling call site program
      location. But the wording suggests that is not the intention.

      Given these issues it is suggested ``DW_OP_entry_value`` is deprecated in
      favor of using the new facities that have well defined semantics and
      implementations.

.. _amdgpu-dwarf-location-description-operations:

Location Description Operations
###############################

This section describes the operations that push location descriptions on the
stack.

General Location Description Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.  ``DW_OP_LLVM_offset`` *New*

    ``DW_OP_LLVM_offset`` pops two stack entries. The first must be an integral
    type value that represents a byte displacement B. The second must be a
    location description L.

    It adds the value of B scaled by 8 (the byte size) to the bit offset of each
    single location description SL of L, and pushes the updated L.

    If the updated bit offset of any SL is less than 0 or greater than or equal
    to the size of the location storage specified by SL, then the DWARF
    expression is ill-formed.

2.  ``DW_OP_LLVM_offset_uconst`` *New*

    ``DW_OP_LLVM_offset_uconst`` has a single unsigned LEB128 integer operand
    that represents a byte displacement B.

    The operation is equivalent to performing ``DW_OP_constu B;
    DW_OP_LLVM_offset``.

    *This operation is supplied specifically to be able to encode more field
    displacements in two bytes than can be done with* ``DW_OP_lit*;
    DW_OP_LLVM_offset``\ *.*

    .. note::

      Should this be named ``DW_OP_LLVM_offset_uconst`` to match
      ``DW_OP_plus_uconst``, or ``DW_OP_LLVM_offset_constu`` to match
      ``DW_OP_constu``?

3.  ``DW_OP_LLVM_bit_offset`` *New*

    ``DW_OP_LLVM_bit_offset`` pops two stack entries. The first must be an
    integral type value that represents a bit displacement B. The second must be
    a location description L.

    It adds the value of B to the bit offset of each single location description
    SL of L, and pushes the updated L.

    If the updated bit offset of any SL is less than 0 or greater than or equal
    to the size of the location storage specified by SL, then the DWARF
    expression is ill-formed.

4.  ``DW_OP_push_object_address``

    ``DW_OP_push_object_address`` pushes the location description L of the
    object currently being evaluated as part of evaluation of a user presented
    expression.

    This object may correspond to an independent variable described by its own
    debugging information entry or it may be a component of an array, structure,
    or class whose address has been dynamically determined by an earlier step
    during user expression evaluation.

    *This operation provides explicit functionality (especially for arrays
    involving descriptions) that is analogous to the implicit push of the base
    location description of a structure prior to evaluation of a
    ``DW_AT_data_member_location`` to access a data member of a structure.*

5.  ``DW_OP_LLVM_call_frame_entry_reg`` *New*

    ``DW_OP_LLVM_call_frame_entry_reg`` has a single unsigned LEB128 integer
    operand that represents a target architecture register number R.

    It pushes a location description L that holds the value of register R on
    entry to the current subprogram as defined by the Call Frame Information
    (see :ref:`amdgpu-dwarf-call-frame-information`).

    *If there is no Call Frame Information defined, then the default rules for
    the target architecture are used. If the register rule is* undefined\ *, then
    the undefined location description is pushed. If the register rule is* same
    value\ *, then a register location description for R is pushed.*

Undefined Location Description Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*The undefined location storage represents a piece or all of an object that is
present in the source but not in the object code (perhaps due to optimization).
Neither reading nor writing to the undefined location storage is meaningful.*

An undefined location description specifies the undefined location storage.
There is no concept of the size of the undefined location storage, nor of a bit
offset for an undefined location description. The ``DW_OP_LLVM_*offset``
operations leave an undefined location description unchanged. The
``DW_OP_*piece`` operations can explicitly or implicitly specify an undefined
location description, allowing any size and offset to be specified, and results
in a part with all undefined bits.

1.  ``DW_OP_LLVM_undefined`` *New*

    ``DW_OP_LLVM_undefined`` pushes a location description L that comprises one
    undefined location description SL.

.. _amdgpu-dwarf-memory-location-description-operations:

Memory Location Description Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each of the target architecture specific address spaces has a corresponding
memory location storage that denotes the linear addressable memory of that
address space. The size of each memory location storage corresponds to the range
of the addresses in the corresponding address space.

*It is target architecture defined how address space location storage maps to
target architecture physical memory. For example, they may be independent
memory, or more than one location storage may alias the same physical memory
possibly at different offsets and with different interleaving. The mapping may
also be dictated by the source language address classes.*

A memory location description specifies a memory location storage. The bit
offset corresponds to a bit position within a byte of the memory. Bits accessed
using a memory location description, access the corresponding target
architecture memory starting at the bit position within the byte specified by
the bit offset.

A memory location description that has a bit offset that is a multiple of 8 (the
byte size) is defined to be a byte address memory location description. It has a
memory byte address A that is equal to the bit offset divided by 8.

A memory location description that does not have a bit offset that is a multiple
of 8 (the byte size) is defined to be a bit field memory location description.
It has a bit position B equal to the bit offset modulo 8, and a memory byte
address A equal to the bit offset minus B that is then divided by 8.

The address space AS of a memory location description is defined to be the
address space that corresponds to the memory location storage associated with
the memory location description.

A location description that is comprised of one byte address memory location
description SL is defined to be a memory byte address location description. It
has a byte address equal to A and an address space equal to AS of the
corresponding SL.

``DW_ASPACE_none`` is defined as the target architecture default address space.

If a stack entry is required to be a location description, but it is a value V
with the generic type, then it is implicitly converted to a location description
L with one memory location description SL. SL specifies the memory location
storage that corresponds to the target architecture default address space with a
bit offset equal to V scaled by 8 (the byte size).

.. note::

  If it is wanted to allow any integral type value to be implicitly converted to
  a memory location description in the target architecture default address
  space:

    If a stack entry is required to be a location description, but is a value V
    with an integral type, then it is implicitly converted to a location
    description L with a one memory location description SL. If the type size of
    V is less than the generic type size, then the value V is zero extended to
    the size of the generic type. The least significant generic type size bits
    are treated as a twos-complement unsigned value to be used as an address A.
    SL specifies memory location storage corresponding to the target
    architecture default address space with a bit offset equal to A scaled by 8
    (the byte size).

  The implicit conversion could also be defined as target architecture specific.
  For example, GDB checks if V is an integral type. If it is not it gives an
  error. Otherwise, GDB zero-extends V to 64 bits. If the GDB target defines a
  hook function, then it is called. The target specific hook function can modify
  the 64-bit value, possibly sign extending based on the original value type.
  Finally, GDB treats the 64-bit value V as a memory location address.

If a stack entry is required to be a location description, but it is an implicit
pointer value IPV with the target architecture default address space, then it is
implicitly converted to a location description with one single location
description specified by IPV. See
:ref:`amdgpu-dwarf-implicit-location-descriptions`.

.. note::

  Is this rule required for DWARF Version 5 backwards compatibility? If not, it
  can be eliminated, and the producer can use
  ``DW_OP_LLVM_form_aspace_address``.

If a stack entry is required to be a value, but it is a location description L
with one memory location description SL in the target architecture default
address space with a bit offset B that is a multiple of 8, then it is implicitly
converted to a value equal to B divided by 8 (the byte size) with the generic
type.

1.  ``DW_OP_addr``

    ``DW_OP_addr`` has a single byte constant value operand, which has the size
    of the generic type, that represents an address A.

    It pushes a location description L with one memory location description SL
    on the stack. SL specifies the memory location storage corresponding to the
    target architecture default address space with a bit offset equal to A
    scaled by 8 (the byte size).

    *If the DWARF is part of a code object, then A may need to be relocated. For
    example, in the ELF code object format, A must be adjusted by the difference
    between the ELF segment virtual address and the virtual address at which the
    segment is loaded.*

2.  ``DW_OP_addrx``

    ``DW_OP_addrx`` has a single unsigned LEB128 integer operand that represents
    a zero-based index into the ``.debug_addr`` section relative to the value of
    the ``DW_AT_addr_base`` attribute of the associated compilation unit. The
    address value A in the ``.debug_addr`` section has the size of the generic
    type.

    It pushes a location description L with one memory location description SL
    on the stack. SL specifies the memory location storage corresponding to the
    target architecture default address space with a bit offset equal to A
    scaled by 8 (the byte size).

    *If the DWARF is part of a code object, then A may need to be relocated. For
    example, in the ELF code object format, A must be adjusted by the difference
    between the ELF segment virtual address and the virtual address at which the
    segment is loaded.*

3.  ``DW_OP_LLVM_form_aspace_address`` *New*

    ``DW_OP_LLVM_form_aspace_address`` pops top two stack entries. The first
    must be an integral type value that represents a target architecture
    specific address space identifier AS. The second must be an integral type
    value that represents an address A.

    The address size S is defined as the address bit size of the target
    architecture specific address space that corresponds to AS.

    A is adjusted to S bits by zero extending if necessary, and then treating the
    least significant S bits as a twos-complement unsigned value A'.

    It pushes a location description L with one memory location description SL
    on the stack. SL specifies the memory location storage that corresponds to
    AS with a bit offset equal to A' scaled by 8 (the byte size).

    The DWARF expression is ill-formed if AS is not one of the values defined by
    the target architecture specific ``DW_ASPACE_*`` values.

    See :ref:`amdgpu-dwarf-implicit-location-descriptions` for special rules
    concerning implicit pointer values produced by dereferencing implicit
    location descriptions created by the ``DW_OP_implicit_pointer`` and
    ``DW_OP_LLVM_implicit_aspace_pointer`` operations.

4.  ``DW_OP_form_tls_address``

    ``DW_OP_form_tls_address`` pops one stack entry that must be an integral
    type value and treats it as a thread-local storage address T.

    It pushes a location description L with one memory location description SL
    on the stack. SL is the target architecture specific memory location
    description that corresponds to the thread-local storage address T.

    The meaning of the thread-local storage address T is defined by the run-time
    environment. If the run-time environment supports multiple thread-local
    storage blocks for a single thread, then the block corresponding to the
    executable or shared library containing this DWARF expression is used.

    *Some implementations of C, C++, Fortran, and other languages support a
    thread-local storage class. Variables with this storage class have distinct
    values and addresses in distinct threads, much as automatic variables have
    distinct values and addresses in each subprogram invocation. Typically,
    there is a single block of storage containing all thread-local variables
    declared in the main executable, and a separate block for the variables
    declared in each shared library. Each thread-local variable can then be
    accessed in its block using an identifier. This identifier is typically a
    byte offset into the block and pushed onto the DWARF stack by one of the*
    ``DW_OP_const*`` *operations prior to the* ``DW_OP_form_tls_address``
    *operation. Computing the address of the appropriate block can be complex
    (in some cases, the compiler emits a function call to do it), and difficult
    to describe using ordinary DWARF location descriptions. Instead of forcing
    complex thread-local storage calculations into the DWARF expressions, the*
    ``DW_OP_form_tls_address`` *allows the consumer to perform the computation
    based on the target architecture specific run-time environment.*

5.  ``DW_OP_call_frame_cfa``

    ``DW_OP_call_frame_cfa`` pushes the location description L of the Canonical
    Frame Address (CFA) of the current subprogram, obtained from the Call Frame
    Information on the stack. See :ref:`amdgpu-dwarf-call-frame-information`.

    *Although the value of the* ``DW_AT_frame_base`` *attribute of the debugger
    information entry corresponding to the current subprogram can be computed
    using a location list expression, in some cases this would require an
    extensive location list because the values of the registers used in
    computing the CFA change during a subprogram execution. If the Call Frame
    Information is present, then it already encodes such changes, and it is
    space efficient to reference that using the* ``DW_OP_call_frame_cfa``
    *operation.*

6.  ``DW_OP_fbreg``

    ``DW_OP_fbreg`` has a single signed LEB128 integer operand that represents a
    byte displacement B.

    The location description L for the *frame base* of the current subprogram is
    obtained from the ``DW_AT_frame_base`` attribute of the debugger information
    entry corresponding to the current subprogram as described in
    :ref:`amdgpu-dwarf-debugging-information-entry-attributes`.

    The location description L is updated as if the ``DW_OP_LLVM_offset_uconst
    B`` operation was applied. The updated L is pushed on the stack.

7.  ``DW_OP_breg0``, ``DW_OP_breg1``, ..., ``DW_OP_breg31``

    The ``DW_OP_breg<N>`` operations encode the numbers of up to 32 registers,
    numbered from 0 through 31, inclusive. The register number R corresponds to
    the N in the operation name.

    They have a single signed LEB128 integer operand that represents a byte
    displacement B.

    The address space identifier AS is defined as the one corresponding to the
    target architecture specific default address space.

    The address size S is defined as the address bit size of the target
    architecture specific address space corresponding to AS.

    The contents of the register specified by R are retrieved as a
    twos-complement unsigned value and zero extended to S bits. B is added and
    the least significant S bits are treated as a twos-complement unsigned value
    to be used as an address A.

    They push a location description L comprising one memory location
    description LS on the stack. LS specifies the memory location storage that
    corresponds to AS with a bit offset equal to A scaled by 8 (the byte size).

8.  ``DW_OP_bregx``

    ``DW_OP_bregx`` has two operands. The first is an unsigned LEB128 integer
    that represents a register number R. The second is a signed LEB128
    integer that represents a byte displacement B.

    The action is the same as for ``DW_OP_breg<N>`` except that R is used as the
    register number and B is used as the byte displacement.

9.  ``DW_OP_LLVM_aspace_bregx`` *New*

    ``DW_OP_LLVM_aspace_bregx`` has two operands. The first is an unsigned
    LEB128 integer that represents a register number R. The second is a signed
    LEB128 integer that represents a byte displacement B. It pops one stack
    entry that is required to be an integral type value that represents a target
    architecture specific address space identifier AS.

    The action is the same as for ``DW_OP_breg<N>`` except that R is used as the
    register number, B is used as the byte displacement, and AS is used as the
    address space identifier.

    The DWARF expression is ill-formed if AS is not one of the values defined by
    the target architecture specific ``DW_ASPACE_*`` values.

    .. note::

      Could also consider adding ``DW_OP_aspace_breg0, DW_OP_aspace_breg1, ...,
      DW_OP_aspace_bref31`` which would save encoding size.

.. _amdgpu-dwarf-register-location-descriptions:

Register Location Description Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is a register location storage that corresponds to each of the target
architecture registers. The size of each register location storage corresponds
to the size of the corresponding target architecture register.

A register location description specifies a register location storage. The bit
offset corresponds to a bit position within the register. Bits accessed using a
register location description access the corresponding target architecture
register starting at the specified bit offset.

1.  ``DW_OP_reg0``, ``DW_OP_reg1``, ..., ``DW_OP_reg31``

    ``DW_OP_reg<N>`` operations encode the numbers of up to 32 registers,
    numbered from 0 through 31, inclusive. The target architecture register
    number R corresponds to the N in the operation name.

    They push a location description L that specifies one register location
    description SL on the stack. SL specifies the register location storage that
    corresponds to R with a bit offset of 0.

2.  ``DW_OP_regx``

    ``DW_OP_regx`` has a single unsigned LEB128 integer operand that represents
    a target architecture register number R.

    It pushes a location description L that specifies one register location
    description SL on the stack. SL specifies the register location storage that
    corresponds to R with a bit offset of 0.

*These operations obtain a register location. To fetch the contents of a
register, it is necessary to use* ``DW_OP_regval_type``\ *, use one of the*
``DW_OP_breg*`` *register-based addressing operations, or use* ``DW_OP_deref*``
*on a register location description.*

.. _amdgpu-dwarf-implicit-location-descriptions:

Implicit Location Description Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Implicit location storage represents a piece or all of an object which has no
actual location in the program but whose contents are nonetheless known, either
as a constant or can be computed from other locations and values in the program.

An implicit location description specifies an implicit location storage. The bit
offset corresponds to a bit position within the implicit location storage. Bits
accessed using an implicit location description, access the corresponding
implicit storage value starting at the bit offset.

1.  ``DW_OP_implicit_value``

    ``DW_OP_implicit_value`` has two operands. The first is an unsigned LEB128
    integer that represents a byte size S. The second is a block of bytes with a
    length equal to S treated as a literal value V.

    An implicit location storage LS is created with the literal value V and a
    size of S.

    It pushes location description L with one implicit location description SL
    on the stack. SL specifies LS with a bit offset of 0.

2.  ``DW_OP_stack_value``

    ``DW_OP_stack_value`` pops one stack entry that must be a value V.

    An implicit location storage LS is created with the literal value V and a
    size equal to V's base type size.

    It pushes a location description L with one implicit location description SL
    on the stack. SL specifies LS with a bit offset of 0.

    *The* ``DW_OP_stack_value`` *operation specifies that the object does not
    exist in memory, but its value is nonetheless known. In this form, the
    location description specifies the actual value of the object, rather than
    specifying the memory or register storage that holds the value.*

    See :ref:`amdgpu-dwarf-implicit-location-descriptions` for special rules
    concerning implicit pointer values produced by dereferencing implicit
    location descriptions created by the ``DW_OP_implicit_pointer`` and
    ``DW_OP_LLVM_implicit_aspace_pointer`` operations.

    .. note::

      Since location descriptions are allowed on the stack, the
      ``DW_OP_stack_value`` operation no longer terminates the DWARF operation
      expression execution as in DWARF Version 5.

3.  ``DW_OP_implicit_pointer``

    *An optimizing compiler may eliminate a pointer, while still retaining the
    value that the pointer addressed.* ``DW_OP_implicit_pointer`` *allows a
    producer to describe this value.*

    ``DW_OP_implicit_pointer`` *specifies an object is a pointer to the target
    architecture default address space that cannot be represented as a real
    pointer, even though the value it would point to can be described. In this
    form, the location description specifies a debugging information entry that
    represents the actual location description of the object to which the
    pointer would point. Thus, a consumer of the debug information would be able
    to access the dereferenced pointer, even when it cannot access the pointer
    itself.*

    ``DW_OP_implicit_pointer`` has two operands. The first is a 4-byte unsigned
    value in the 32-bit DWARF format, or an 8-byte unsigned value in the 64-bit
    DWARF format, that represents a debugging information entry reference R. The
    second is a signed LEB128 integer that represents a byte displacement B.

    R is used as the offset of a debugging information entry D in a
    ``.debug_info`` section, which may be contained in an executable or shared
    object file other than that containing the operation. For references from one
    executable or shared object file to another, the relocation must be
    performed by the consumer.

    *The first operand interpretation is exactly like that for*
    ``DW_FORM_ref_addr``\ *.*

    The address space identifier AS is defined as the one corresponding to the
    target architecture specific default address space.

    The address size S is defined as the address bit size of the target
    architecture specific address space corresponding to AS.

    An implicit location storage LS is created with the debugging information
    entry D, address space AS, and size of S.

    It pushes a location description L that comprises one implicit location
    description SL on the stack. SL specifies LS with a bit offset of 0.

    If a ``DW_OP_deref*`` operation pops a location description L', and
    retrieves S bits where both:

    1.  All retrieved bits come from an implicit location description that
        refers to an implicit location storage that is the same as LS.

        *Note that all bits do not have to come from the same implicit location
        description, as L' may involve composite location descriptors.*

    2.  The bits come from consecutive ascending offsets within their respective
        implicit location storage.

    *These rules are equivalent to retrieving the complete contents of LS.*

    Then the value V pushed by the ``DW_OP_deref*`` operation is an implicit
    pointer value IPV with a target architecture specific address space of AS, a
    debugging information entry of D, and a base type of T. If AS is the target
    architecture default address space, then T is the generic type. Otherwise, T
    is a target architecture specific integral type with a bit size equal to S.

    Otherwise, if a ``DW_OP_deref*`` operation is applied to a location
    description such that some retrieved bits come from an implicit location
    storage that is the same as LS, then the DWARF expression is ill-formed.

    If IPV is either implicitly converted to a location description (only done
    if AS is the target architecture default address space) or used by
    ``DW_OP_LLVM_form_aspace_address`` (only done if the address space specified
    is AS), then the resulting location description RL is:

    * If D has a ``DW_AT_location`` attribute, the DWARF expression E from the
      ``DW_AT_location`` attribute is evaluated as a location description. The
      current subprogram and current program location of the evaluation context
      that is accessing IPV is used for the evaluation context of E, together
      with an empty initial stack. RL is the expression result.

    * If D has a ``DW_AT_const_value`` attribute, then an implicit location
      storage RLS is created from the ``DW_AT_const_value`` attribute's value
      with a size matching the size of the ``DW_AT_const_value`` attribute's
      value. RL comprises one implicit location description SRL. SRL specifies
      RLS with a bit offset of 0.

      .. note::

        If using ``DW_AT_const_value`` for variables and formal parameters is
        deprecated and instead ``DW_AT_location`` is used with an implicit
        location description, then this rule would not be required.

    * Otherwise the DWARF expression is ill-formed.

    The bit offset of RL is updated as if the ``DW_OP_LLVM_offset_uconst B``
    operation was applied.

    If a ``DW_OP_stack_value`` operation pops a value that is the same as IPV,
    then it pushes a location description that is the same as L.

    The DWARF expression is ill-formed if it accesses LS or IPV in any other
    manner.

    *The restrictions on how an implicit pointer location description created
    by* ``DW_OP_implicit_pointer`` *and* ``DW_OP_LLVM_aspace_implicit_pointer``
    *can be used are to simplify the DWARF consumer. Similarly, for an implicit
    pointer value created by* ``DW_OP_deref*`` *and* ``DW_OP_stack_value``\ .*

4.  ``DW_OP_LLVM_aspace_implicit_pointer`` *New*

    ``DW_OP_LLVM_aspace_implicit_pointer`` has two operands that are the same as
    for ``DW_OP_implicit_pointer``.

    It pops one stack entry that must be an integral type value that represents
    a target architecture specific address space identifier AS.

    The location description L that is pushed on the stack is the same as for
    ``DW_OP_implicit_pointer`` except that the address space identifier used is
    AS.

    The DWARF expression is ill-formed if AS is not one of the values defined by
    the target architecture specific ``DW_ASPACE_*`` values.

*Typically a* ``DW_OP_implicit_pointer`` *or*
``DW_OP_LLVM_aspace_implicit_pointer`` *operation is used in a DWARF expression
E*\ :sub:`1` *of a* ``DW_TAG_variable`` *or* ``DW_TAG_formal_parameter``
*debugging information entry D*\ :sub:`1`\ *'s* ``DW_AT_location`` *attribute.
The debugging information entry referenced by the* ``DW_OP_implicit_pointer``
*or* ``DW_OP_LLVM_aspace_implicit_pointer`` *operations is typically itself a*
``DW_TAG_variable`` *or* ``DW_TAG_formal_parameter`` *debugging information
entry D*\ :sub:`2` *whose* ``DW_AT_location`` *attribute gives a second DWARF
expression E*\ :sub:`2`\ *.*

*D*\ :sub:`1` *and E*\ :sub:`1` *are describing the location of a pointer type
object. D*\ :sub:`2` *and E*\ :sub:`2` *are describing the location of the
object pointed to by that pointer object.*

*However, D*\ :sub:`2` *may be any debugging information entry that contains a*
``DW_AT_location`` *or* ``DW_AT_const_value`` *attribute (for example,*
``DW_TAG_dwarf_procedure``\ *). By using E*\ :sub:`2`\ *, a consumer can
reconstruct the value of the object when asked to dereference the pointer
described by E*\ :sub:`1` *which contains the* ``DW_OP_implicit_pointer`` or
``DW_OP_LLVM_aspace_implicit_pointer`` *operation.*

.. _amdgpu-dwarf-composite-location-description-operations:

Composite Location Description Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A composite location storage represents an object or value which may be
contained in part of another location storage or contained in parts of more
than one location storage.

Each part has a part location description L and a part bit size S. L can have
one or more single location descriptions SL. If there are more than one SL then
that indicates that part is located in more than one place. The bits of each
place of the part comprise S contiguous bits from the location storage LS
specified by SL starting at the bit offset specified by SL. All the bits must
be within the size of LS or the DWARF expression is ill-formed.

A composite location storage can have zero or more parts. The parts are
contiguous such that the zero-based location storage bit index will range over
each part with no gaps between them. Therefore, the size of a composite location
storage is the sum of the size of its parts. The DWARF expression is ill-formed
if the size of the contiguous location storage is larger than the size of the
memory location storage corresponding to the largest target architecture
specific address space.

A composite location description specifies a composite location storage. The bit
offset corresponds to a bit position within the composite location storage.

There are operations that create a composite location storage.

There are other operations that allow a composite location storage to be
incrementally created. Each part is created by a separate operation. There may
be one or more operations to create the final composite location storage. A
series of such operations describes the parts of the composite location storage
that are in the order that the associated part operations are executed.

To support incremental creation, a composite location storage can be in an
incomplete state. When an incremental operation operates on an incomplete
composite location storage, it adds a new part, otherwise it creates a new
composite location storage. The ``DW_OP_LLVM_piece_end`` operation explicitly
makes an incomplete composite location storage complete.

A composite location description that specifies a composite location storage
that is incomplete is termed an incomplete composite location description. A
composite location description that specifies a composite location storage that
is complete is termed a complete composite location description.

If the top stack entry is a location description that has one incomplete
composite location description SL after the execution of an operation expression
has completed, SL is converted to a complete composite location description.

*Note that this conversion does not happen after the completion of an operation
expression that is evaluated on the same stack by the* ``DW_OP_call*``
*operations. Such executions are not a separate evaluation of an operation
expression, but rather the continued evaluation of the same operation expression
that contains the* ``DW_OP_call*`` *operation.*

If a stack entry is required to be a location description L, but L has an
incomplete composite location description, then the DWARF expression is
ill-formed. The exception is for the operations involved in incrementally
creating a composite location description as described below.

*Note that a DWARF operation expression may arbitrarily compose composite
location descriptions from any other location description, including those that
have multiple single location descriptions, and those that have composite
location descriptions.*

*The incremental composite location description operations are defined to be
compatible with the definitions in DWARF Version 5.*

1.  ``DW_OP_piece``

    ``DW_OP_piece`` has a single unsigned LEB128 integer that represents a byte
    size S.

    The action is based on the context:

    * If the stack is empty, then a location description L comprised of one
      incomplete composite location description SL is pushed on the stack.

      An incomplete composite location storage LS is created with a single part
      P. P specifies a location description PL and has a bit size of S scaled by
      8 (the byte size). PL is comprised of one undefined location description
      PSL.

      SL specifies LS with a bit offset of 0.

    * Otherwise, if the top stack entry is a location description L comprised of
      one incomplete composite location description SL, then the incomplete
      composite location storage LS that SL specifies is updated to append a new
      part P. P specifies a location description PL and has a bit size of S
      scaled by 8 (the byte size). PL is comprised of one undefined location
      description PSL. L is left on the stack.

    * Otherwise, if the top stack entry is a location description or can be
      converted to one, then it is popped and treated as a part location
      description PL. Then:

      * If the top stack entry (after popping PL) is a location description L
        comprised of one incomplete composite location description SL, then the
        incomplete composite location storage LS that SL specifies is updated to
        append a new part P. P specifies the location description PL and has a
        bit size of S scaled by 8 (the byte size). L is left on the stack.

      * Otherwise, a location description L comprised of one incomplete
        composite location description SL is pushed on the stack.

        An incomplete composite location storage LS is created with a single
        part P. P specifies the location description PL and has a bit size of S
        scaled by 8 (the byte size).

        SL specifies LS with a bit offset of 0.

    * Otherwise, the DWARF expression is ill-formed

    *Many compilers store a single variable in sets of registers or store a
    variable partially in memory and partially in registers.* ``DW_OP_piece``
    *provides a way of describing where a part of a variable is located.*

    *If a non-0 byte displacement is required, the* ``DW_OP_LLVM_offset``
    *operation can be used to update the location description before using it as
    the part location description of a* ``DW_OP_piece`` *operation.*

    *The evaluation rules for the* ``DW_OP_piece`` *operation allow it to be
    compatible with the DWARF Version 5 definition.*

    .. note::

      Since this proposal allows location descriptions to be entries on the
      stack, a simpler operation to create composite location descriptions. For
      example, just one operation that specifies how many parts, and pops pairs
      of stack entries for the part size and location description. Not only
      would this be a simpler operation and avoid the complexities of incomplete
      composite location descriptions, but it may also have a smaller encoding
      in practice. However, the desire for compatibility with DWARF Version 5 is
      likely a stronger consideration.

2.  ``DW_OP_bit_piece``

    ``DW_OP_bit_piece`` has two operands. The first is an unsigned LEB128
    integer that represents the part bit size S. The second is an unsigned
    LEB128 integer that represents a bit displacement B.

    The action is the same as for ``DW_OP_piece`` except that any part created
    has the bit size S, and the location description PL of any created part is
    updated as if the ``DW_OP_constu B; DW_OP_LLVM_bit_offset`` operations were
    applied.

    ``DW_OP_bit_piece`` *is used instead of* ``DW_OP_piece`` *when the piece to
    be assembled is not byte-sized or is not at the start of the part location
    description.*

    *If a computed bit displacement is required, the* ``DW_OP_LLVM_bit_offset``
    *operation can be used to update the location description before using it as
    the part location description of a* ``DW_OP_bit_piece`` *operation.*

    .. note::

      The bit offset operand is not needed as ``DW_OP_LLVM_bit_offset`` can be
      used on the part's location description.

3.  ``DW_OP_LLVM_piece_end`` *New*

    If the top stack entry is not a location description L comprised of one
    incomplete composite location description SL, then the DWARF expression is
    ill-formed.

    Otherwise, the incomplete composite location storage LS specified by SL is
    updated to be a complete composite location description with the same parts.

4.  ``DW_OP_LLVM_extend`` *New*

    ``DW_OP_LLVM_extend`` has two operands. The first is an unsigned LEB128
    integer that represents the element bit size S. The second is an unsigned
    LEB128 integer that represents a count C.

    It pops one stack entry that must be a location description and is treated
    as the part location description PL.

    A location description L comprised of one complete composite location
    description SL is pushed on the stack.

    A complete composite location storage LS is created with C identical parts
    P. Each P specifies PL and has a bit size of S.

    SL specifies LS with a bit offset of 0.

    The DWARF expression is ill-formed if the element bit size or count are 0.

5.  ``DW_OP_LLVM_select_bit_piece`` *New*

    ``DW_OP_LLVM_select_bit_piece`` has two operands. The first is an unsigned
    LEB128 integer that represents the element bit size S. The second is an
    unsigned LEB128 integer that represents a count C.

    It pops three stack entries. The first must be an integral type value that
    represents a bit mask value M. The second must be a location description
    that represents the one-location description L1. The third must be a
    location description that represents the zero-location description L0.

    A complete composite location storage LS is created with C parts P\ :sub:`N`
    ordered in ascending N from 0 to C-1 inclusive. Each P\ :sub:`N` specifies
    location description PL\ :sub:`N` and has a bit size of S.

    PL\ :sub:`N` is as if the ``DW_OP_LLVM_bit_offset N*S`` operation was
    applied to PLX\ :sub:`N`\ .

    PLX\ :sub:`N` is the same as L0 if the N\ :sup:`th` least significant bit of
    M is a zero, otherwise it is the same as L1.

    A location description L comprised of one complete composite location
    description SL is pushed on the stack. SL specifies LS with a bit offset of
    0.

    The DWARF expression is ill-formed if S or C are 0, or if the bit size of M
    is less than C.

.. _amdgpu-dwarf-location-list-expressions:

DWARF Location List Expressions
+++++++++++++++++++++++++++++++

*To meet the needs of recent computer architectures and optimization techniques,
debugging information must be able to describe the location of an object whose
location changes over the object’s lifetime, and may reside at multiple
locations during parts of an object's lifetime. Location list expressions are
used in place of operation expressions whenever the object whose location is
being described has these requirements.*

A location list expression consists of a series of location list entries. Each
location list entry is one of the following kinds:

*Bounded location description*

  This kind of location list entry provides an operation expression that
  evaluates to the location description of an object that is valid over a
  lifetime bounded by a starting and ending address. The starting address is the
  lowest address of the address range over which the location is valid. The
  ending address is the address of the first location past the highest address
  of the address range.

  The location list entry matches when the current program location is within
  the given range.

  There are several kinds of bounded location description entries which differ
  in the way that they specify the starting and ending addresses.

*Default location description*

  This kind of location list entry provides an operation expression that
  evaluates to the location description of an object that is valid when no
  bounded location description entry applies.

  The location list entry matches when the current program location is not
  within the range of any bounded location description entry.

*Base address*

  This kind of location list entry provides an address to be used as the base
  address for beginning and ending address offsets given in certain kinds of
  bounded location description entries. The applicable base address of a bounded
  location description entry is the address specified by the closest preceding
  base address entry in the same location list. If there is no preceding base
  address entry, then the applicable base address defaults to the base address
  of the compilation unit (see DWARF Version 5 section 3.1.1).

  In the case of a compilation unit where all of the machine code is contained
  in a single contiguous section, no base address entry is needed.

*End-of-list*

  This kind of location list entry marks the end of the location list
  expression.

The address ranges defined by the bounded location description entries of a
location list expression may overlap. When they do, they describe a situation in
which an object exists simultaneously in more than one place.

If all of the address ranges in a given location list expression do not
collectively cover the entire range over which the object in question is
defined, and there is no following default location description entry, it is
assumed that the object is not available for the portion of the range that is
not covered.

The operation expression of each matching location list entry is evaluated as a
location description and its result is returned as the result of the location
list entry. The operation expression is evaluated with the same context as the
location list expression, including the same current frame, current program
location, and initial stack.

The result of the evaluation of a DWARF location list expression is a location
description that is comprised of the union of the single location descriptions
of the location description result of each matching location list entry. If
there are no matching location list entries, then the result is a location
description that comprises one undefined location description.

A location list expression can only be used as the value of a debugger
information entry attribute that is encoded using class ``loclist`` or
``loclistsptr`` (see DWARF Version 5 section 7.5.5). The value of the attribute
provides an index into a separate object file section called ``.debug_loclists``
or ``.debug_loclists.dwo`` (for split DWARF object files) that contains the
location list entries.

A ``DW_OP_call*`` and ``DW_OP_implicit_pointer`` operation can be used to
specify a debugger information entry attribute that has a location list
expression. Several debugger information entry attributes allow DWARF
expressions that are evaluated with an initial stack that includes a location
description that may originate from the evaluation of a location list
expression.

*This location list representation, the* ``loclist`` *and* ``loclistsptr``
*class, and the related* ``DW_AT_loclists_base`` *attribute are new in DWARF
Version 5. Together they eliminate most, or all of the code object relocations
previously needed for location list expressions.*

.. note::

  The rest of this section is the same as DWARF Version 5 section 2.6.2.

.. _amdgpu-dwarf-segment_addresses:

Segmented Addresses
~~~~~~~~~~~~~~~~~~~

.. note::

  This augments DWARF Version 5 section 2.12.

DWARF address classes are used for source languages that have the concept of
memory spaces. They are used in the ``DW_AT_address_class`` attribute for
pointer type, reference type, subprogram, and subprogram type debugger
information entries.

Each DWARF address class is conceptually a separate source language memory space
with its own lifetime and aliasing rules. DWARF address classes are used to
specify the source language memory spaces that pointer type and reference type
values refer, and to specify the source language memory space in which variables
are allocated.

The set of currently defined source language DWARF address classes, together
with source language mappings, is given in
:ref:`amdgpu-dwarf-address-class-table`.

Vendor defined source language address classes may be defined using codes in the
range ``DW_ADDR_LLVM_lo_user`` to ``DW_ADDR_LLVM_hi_user``.

.. table:: Address class
   :name: amdgpu-dwarf-address-class-table

   ========================= ============ ========= ========= =========
   Address Class Name        Meaning      C/C++     OpenCL    CUDA/HIP
   ========================= ============ ========= ========= =========
   ``DW_ADDR_none``          generic      *default* generic   *default*
   ``DW_ADDR_LLVM_global``   global                 global
   ``DW_ADDR_LLVM_constant`` constant               constant  constant
   ``DW_ADDR_LLVM_group``    thread-group           local     shared
   ``DW_ADDR_LLVM_private``  thread                 private
   ``DW_ADDR_LLVM_lo_user``
   ``DW_ADDR_LLVM_hi_user``
   ========================= ============ ========= ========= =========

DWARF address spaces correspond to target architecture specific linear
addressable memory areas. They are used in DWARF expression location
descriptions to describe in which target architecture specific memory area data
resides.

*Target architecture specific DWARF address spaces may correspond to hardware
supported facilities such as memory utilizing base address registers, scratchpad
memory, and memory with special interleaving. The size of addresses in these
address spaces may vary. Their access and allocation may be hardware managed
with each thread or group of threads having access to independent storage. For
these reasons they may have properties that do not allow them to be viewed as
part of the unified global virtual address space accessible by all threads.*

*It is target architecture specific whether multiple DWARF address spaces are
supported and how source language DWARF address classes map to target
architecture specific DWARF address spaces. A target architecture may map
multiple source language DWARF address classes to the same target architecture
specific DWARF address class. Optimization may determine that variable lifetime
and access pattern allows them to be allocated in faster scratchpad memory
represented by a different DWARF address space.*

Although DWARF address space identifiers are target architecture specific,
``DW_ASPACE_none`` is a common address space supported by all target
architectures.

DWARF address space identifiers are used by:

* The DWARF expession operations: ``DW_OP_LLVM_aspace_bregx``,
  ``DW_OP_LLVM_form_aspace_address``, ``DW_OP_LLVM_implicit_aspace_pointer``,
  and ``DW_OP_xderef*``.

* The CFI instructions: ``DW_CFA_def_aspace_cfa`` and
  ``DW_CFA_def_aspace_cfa_sf``.

.. note::

  With the definition of DWARF address classes and DWARF address spaces in this
  proposal, DWARF Version 5 table 2.7 needs to be updated. It seems it is an
  example of DWARF address spaces and not DWARF address classes.

.. note::

  With the expanded support for DWARF address spaces in this proposal, it may be
  worth examining if DWARF segments can be eliminated and DWARF address spaces
  used instead.

  That may involve extending DWARF address spaces to also be used to specify
  code locations. In target architectures that use different memory areas for
  code and data this would seem a natural use for DWARF address spaces. This
  would allow DWARF expression location descriptions to be used to describe the
  location of subprograms and entry points that are used in expressions
  involving subprogram pointer type values.

  Currently, DWARF expressions assume data and code resides in the same default
  DWARF address space, and only the address ranges in DWARF location list
  entries and in the ``.debug_aranges`` section for accelerated access for
  addresses allow DWARF segments to be used to distinguish.

.. note::

  Currently, DWARF defines address class values as being target architecture
  specific. It is unclear how language specific memory spaces are intended to be
  represented in DWARF using these.

  For example, OpenCL defines memory spaces (called address spaces in OpenCL)
  for ``global``, ``local``, ``constant``, and ``private``. These are part of
  the type system and are modifiers to pointer types. In addition, OpenCL
  defines ``generic`` pointers that can reference either the ``global``,
  ``local``, or ``private`` memory spaces. To support the OpenCL language the
  debugger would want to support casting pointers between the ``generic`` and
  other memory spaces, querying what memory space a ``generic`` pointer value is
  currently referencing, and possibly using pointer casting to form an address
  for a specific memory space out of an integral value.

  The method to use to dereference a pointer type or reference type value is
  defined in DWARF expressions using ``DW_OP_xderef*`` which uses a target
  architecture specific address space.

  DWARF defines the ``DW_AT_address_class`` attribute on pointer type and
  reference type debugger information entries. It specifies the method to use to
  dereference them. Why is the value of this not the same as the address space
  value used in ``DW_OP_xderef*``? In both cases it is target architecture
  specific and the architecture presumably will use the same set of methods to
  dereference pointers in both cases.

  Since ``DW_AT_address_class`` uses a target architecture specific value, it
  cannot in general capture the source language memory space type modifier
  concept. On some architectures all source language memory space modifiers may
  actually use the same method for dereferencing pointers.

  One possibility is for DWARF to add an ``DW_TAG_LLVM_address_class_type``
  debugger information entry type modifier that can be applied to a pointer type
  and reference type. The ``DW_AT_address_class`` attribute could be re-defined
  to not be target architecture specific and instead define generalized language
  values (as is proposed above for DWARF address classes in the table
  :ref:`amdgpu-dwarf-address-class-table`) that will support OpenCL and other
  languages using memory spaces. The ``DW_AT_address_class`` attribute could be
  defined to not be applied to pointer types or reference types, but instead
  only to the new ``DW_TAG_LLVM_address_class_type`` type modifier debugger
  information entry.

  If a pointer type or reference type is not modified by
  ``DW_TAG_LLVM_address_class_type`` or if ``DW_TAG_LLVM_address_class_type``
  has no ``DW_AT_address_class`` attribute, then the pointer type or reference
  type would be defined to use the ``DW_ADDR_none`` address class as currently.
  Since modifiers can be chained, it would need to be defined if multiple
  ``DW_TAG_LLVM_address_class_type`` modifiers were legal, and if so if the
  outermost one is the one that takes precedence.

  A target architecture implementation that supports multiple address spaces
  would need to map ``DW_ADDR_none`` appropriately to support CUDA-like
  languages that have no address classes in the type system but do support
  variable allocation in address classes. Such variable allocation would result
  in the variable's location description needing an address space.

  The approach proposed in :ref:`amdgpu-dwarf-address-class-table` is to define
  the default ``DW_ADDR_none`` to be the generic address class and not the
  global address class. This matches how CLANG and LLVM have added support for
  CUDA-like languages on top of existing C++ language support. This allows all
  addresses to be generic by default which matches CUDA-like languages.

  An alternative approach is to define ``DW_ADDR_none`` as being the global
  address class and then change ``DW_ADDR_LLVM_global`` to
  ``DW_ADDR_LLVM_generic``. This would match the reality that languages that do
  not support multiple memory spaces only have one default global memory space.
  Generally, in these languages if they expose that the target architecture
  supports multiple address spaces, the default one is still the global memory
  space. Then a language that does support multiple memory spaces has to
  explicitly indicate which pointers have the added ability to reference more
  than the global memory space. However, compilers generating DWARF for
  CUDA-like languages would then have to define every CUDA-like language pointer
  type or reference type using ``DW_TAG_LLVM_address_class_type`` with a
  ``DW_AT_address_class`` attribute of ``DW_ADDR_LLVM_generic`` to match the
  language semantics.

  A new ``DW_AT_LLVM_address_space`` attribute could be defined that can be
  applied to pointer type, reference type, subprogram, and subprogram type to
  describe how objects having the given type are dereferenced or called (the
  role that ``DW_AT_address_class`` currently provides). The values of
  ``DW_AT_address_space`` would be target architecture specific and the same as
  used in ``DW_OP_xderef*``.

.. _amdgpu-dwarf-debugging-information-entry-attributes:

Debugging Information Entry Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

  This section provides changes to existing debugger information entry
  attributes and defines attributes added by the proposal. These would be
  incorporated into the appropriate DWARF Version 5 chapter 2 sections.

1.  ``DW_AT_location``

    Any debugging information entry describing a data object (which includes
    variables and parameters) or common blocks may have a ``DW_AT_location``
    attribute, whose value is a DWARF expression E.

    The result of the attribute is obtained by evaluating E as a location
    description in the context of the current subprogram, current program
    location, and with an empty initial stack. See
    :ref:`amdgpu-dwarf-expressions`.

    See :ref:`amdgpu-dwarf-control-flow-operations` for special evaluation rules
    used by the ``DW_OP_call*`` operations.

    .. note::

      Delete the description of how the ``DW_OP_call*`` operations evaluate a
      ``DW_AT_location`` attribute as that is now described in the operations.

    .. note::

      See the discussion about the ``DW_AT_location`` attribute in the
      ``DW_OP_call*`` operation. Having each attribute only have a single
      purpose and single execution semantics seems desirable. It makes it easier
      for the consumer that no longer have to track the context. It makes it
      easier for the producer as it can rely on a single semantics for each
      attribute.

      For that reason, limiting the ``DW_AT_location`` attribute to only
      supporting evaluating the location description of an object, and using a
      different attribute and encoding class for the evaluation of DWARF
      expression *procedures* on the same operation expression stack seems
      desirable.

2.  ``DW_AT_const_value``

    .. note::

      Could deprecate using the ``DW_AT_const_value`` attribute for
      ``DW_TAG_variable`` or ``DW_TAG_formal_parameter`` debugger information
      entries that have been optimized to a constant. Instead,
      ``DW_AT_location`` could be used with a DWARF expression that produces an
      implicit location description now that any location description can be
      used within a DWARF expression. This allows the ``DW_OP_call*`` operations
      to be used to push the location description of any variable regardless of
      how it is optimized.

3.  ``DW_AT_frame_base``

    A ``DW_TAG_subprogram`` or ``DW_TAG_entry_point`` debugger information entry
    may have a ``DW_AT_frame_base`` attribute, whose value is a DWARF expression
    E.

    The result of the attribute is obtained by evaluating E as a location
    description in the context of the current subprogram, current program
    location, and with an empty initial stack.

    The DWARF is ill-formed if E contains an ``DW_OP_fbreg`` operation, or the
    resulting location description L is not comprised of one single location
    description SL.

    If SL a register location description for register R, then L is replaced
    with the result of evaluating a ``DW_OP_bregx R, 0`` operation. This
    computes the frame base memory location description in the target
    architecture default address space.

    *This allows the more compact* ``DW_OPreg*`` *to be used instead of*
    ``DW_OP_breg* 0``\ *.*

    .. note::

      This rule could be removed and require the producer to create the required
      location description directly using ``DW_OP_call_frame_cfa``,
      ``DW_OP_breg*``, or ``DW_OP_LLVM_aspace_bregx``. This would also then
      allow a target to implement the call frames within a large register.

    Otherwise, the DWARF is ill-formed if SL is not a memory location
    description in any of the target architecture specific address spaces.

    The resulting L is the *frame base* for the subprogram or entry point.

    *Typically, E will use the* ``DW_OP_call_frame_cfa`` *operation or be a
    stack pointer register plus or minus some offset.*

4.  ``DW_AT_data_member_location``

    For a ``DW_AT_data_member_location`` attribute there are two cases:

    1.  If the attribute is an integer constant B, it provides the offset in
        bytes from the beginning of the containing entity.

        The result of the attribute is obtained by evaluating a
        ``DW_OP_LLVM_offset B`` operation with an initial stack comprising the
        location description of the beginning of the containing entity.  The
        result of the evaluation is the location description of the base of the
        member entry.

        *If the beginning of the containing entity is not byte aligned, then the
        beginning of the member entry has the same bit displacement within a
        byte.*

    2.  Otherwise, the attribute must be a DWARF expression E which is evaluated
        with a context of the current frame, current program location, and an
        initial stack comprising the location description of the beginning of
        the containing entity. The result of the evaluation is the location
        description of the base of the member entry.

    .. note::

      The beginning of the containing entity can now be any location
      description, including those with more than one single location
      description, and those with single location descriptions that are of any
      kind and have any bit offset.

5.  ``DW_AT_use_location``

    The ``DW_TAG_ptr_to_member_type`` debugging information entry has a
    ``DW_AT_use_location`` attribute whose value is a DWARF expression E. It is
    used to compute the location description of the member of the class to which
    the pointer to member entry points.

    *The method used to find the location description of a given member of a
    class, structure, or union is common to any instance of that class,
    structure, or union and to any instance of the pointer to member type. The
    method is thus associated with the pointer to member type, rather than with
    each object that has a pointer to member type.*

    The ``DW_AT_use_location`` DWARF expression is used in conjunction with the
    location description for a particular object of the given pointer to member
    type and for a particular structure or class instance.

    The result of the attribute is obtained by evaluating E as a location
    description with the context of the current subprogram, current program
    location, and an initial stack comprising two entries. The first entry is
    the value of the pointer to member object itself. The second entry is the
    location description of the base of the entire class, structure, or union
    instance containing the member whose location is being calculated.

6.  ``DW_AT_data_location``

    The ``DW_AT_data_location`` attribute may be used with any type that
    provides one or more levels of hidden indirection and/or run-time parameters
    in its representation. Its value is a DWARF operation expression E which
    computes the location description of the data for an object. When this
    attribute is omitted, the location description of the data is the same as
    the location description of the object.

    The result of the attribute is obtained by evaluating E as a location
    description with the context of the current subprogram, current program
    location, and an empty initial stack.

    *E will typically involve an operation expression that begins with a*
    ``DW_OP_push_object_address`` *operation which loads the location
    description of the object which can then serve as a description in
    subsequent calculation.*

    .. note::

      Since ``DW_AT_data_member_location``, ``DW_AT_use_location``, and
      ``DW_AT_vtable_elem_location`` allow both operation expressions and
      location list expressions, why does ``DW_AT_data_location`` not allow
      both? In all cases they apply to data objects so less likely that
      optimization would cause different operation expressions for different
      program location ranges. But if supporting for some then should be for
      all.

      It seems odd this attribute is not the same as
      ``DW_AT_data_member_location`` in having an initial stack with the
      location description of the object since the expression has to need it.

7.  ``DW_AT_vtable_elem_location``

    An entry for a virtual function also has a ``DW_AT_vtable_elem_location``
    attribute whose value is a DWARF expression E.

    The result of the attribute is obtained by evaluating E as a location
    description with the context of the current subprogram, current program
    location, and an initial stack comprising the location description of the
    object of the enclosing type.

    The resulting location description is the slot for the function within the
    virtual function table for the enclosing class.

8.  ``DW_AT_static_link``

    If a ``DW_TAG_subprogram`` or ``DW_TAG_entry_point`` debugger information
    entry is lexically nested, it may have a ``DW_AT_static_link`` attribute,
    whose value is a DWARF expression E.

    The result of the attribute is obtained by evaluating E as a location
    description with the context of the current subprogram, current program
    location, and an empty initial stack.

    The DWARF is ill-formed if the resulting location description L is is not
    comprised of one memory location description in any of the target
    architecture specific address spaces.

    The resulting L is the *frame base* of the relevant instance of the
    subprogram that immediately lexically encloses the subprogram or entry
    point.

9.  ``DW_AT_return_addr``

    A ``DW_TAG_subprogram``, ``DW_TAG_inlined_subroutine``, or
    ``DW_TAG_entry_point`` debugger information entry may have a
    ``DW_AT_return_addr`` attribute, whose value is a DWARF expression E.

    The result of the attribute is obtained by evaluating E as a location
    description with the context of the current subprogram, current program
    location, and an empty initial stack.

    The DWARF is ill-formed if the resulting location description L is not
    comprised one memory location description in any of the target architecture
    specific address spaces.

    The resulting L is the place where the return address for the subprogram or
    entry point is stored.

    .. note::

      It is unclear why ``DW_TAG_inlined_subroutine`` has a
      ``DW_AT_return_addr`` attribute but not a ``DW_AT_frame_base`` or
      ``DW_AT_static_link`` attribute. Seems it would either have all of them or
      none. Since inlined subprograms do not have a frame it seems they would
      have none of these attributes.

10. ``DW_AT_call_value``, ``DW_AT_call_data_location``, and ``DW_AT_call_data_value``

    A ``DW_TAG_call_site_parameter`` debugger information entry may have a
    ``DW_AT_call_value`` attribute, whose value is a DWARF operation expression
    E\ :sub:`1`\ .

    The result of the ``DW_AT_call_value`` attribute is obtained by evaluating
    E\ :sub:`1` as a value with the context of the call site subprogram, call
    site program location, and an empty initial stack.

    The call site subprogram is the subprogram containing the
    ``DW_TAG_call_site_parameter`` debugger information entry. The call site
    program location is the location of call site in the call site subprogram.

    *The consumer may have to virtually unwind to the call site in order to
    evaluate the attribute. This will provide both the call site subprogram and
    call site program location needed to evaluate the expression.*

    The resulting value V\ :sub:`1` is the value of the parameter at the time of
    the call made by the call site.

    For parameters passed by reference, where the code passes a pointer to a
    location which contains the parameter, or for reference type parameters, the
    ``DW_TAG_call_site_parameter`` debugger information entry may also have a
    ``DW_AT_call_data_location`` attribute whose value is a DWARF operation
    expression E\ :sub:`2`\ , and a ``DW_AT_call_data_value`` attribute whose
    value is a DWARF operation expression E\ :sub:`3`\ .

    The value of the ``DW_AT_call_data_location`` attribute is obtained by
    evaluating E\ :sub:`2` as a location description with the context of the
    call site subprogram, call site program location, and an empty initial
    stack.

    The resulting location description L\ :sub:`2` is the location where the
    referenced parameter lives during the call made by the call site. If E\
    :sub:`2` would just be a ``DW_OP_push_object_address``, then the
    ``DW_AT_call_data_location`` attribute may be omitted.

    The value of the ``DW_AT_call_data_value`` attribute is obtained by
    evaluating E\ :sub:`3` as a value with the context of the call site
    subprogram, call site program location, and an empty initial stack.

    The resulting value V\ :sub:`3` is the value in L\ :sub:`2` at the time of
    the call made by the call site.

    If it is not possible to avoid the expressions of these attributes from
    accessing registers or memory locations that might be clobbered by the
    subprogram being called by the call site, then the associated attribute
    should not be provided.

    *The reason for the restriction is that the parameter may need to be
    accessed during the execution of the callee. The consumer may virtually
    unwind from the called subprogram back to the caller and then evaluate the
    attribute expressions. The call frame information (see*
    :ref:`amdgpu-dwarf-call-frame-information`\ *) will not be able to restore
    registers that have been clobbered, and clobbered memory will no longer have
    the value at the time of the call.*

11. ``DW_AT_LLVM_lanes`` *New*

    For languages that are implemented using a SIMD or SIMT execution model, a
    ``DW_TAG_subprogram``, ``DW_TAG_inlined_subroutine``, or
    ``DW_TAG_entry_point`` debugger information entry may have a
    ``DW_AT_LLVM_lanes`` attribute whose value is an integer constant that is
    the number of lanes per thread. This is the static number of lanes per
    thread. It is not the dynamic number of lanes with which the thread was
    initiated, for example, due to smaller or partial work-groups.

    If not present, the default value of 1 is used.

    The DWARF is ill-formed if the value is 0.

12. ``DW_AT_LLVM_lane_pc`` *New*

    For languages that are implemented using a SIMD or SIMT execution model, a
    ``DW_TAG_subprogram``, ``DW_TAG_inlined_subroutine``, or
    ``DW_TAG_entry_point`` debugging information entry may have a
    ``DW_AT_LLVM_lane_pc`` attribute whose value is a DWARF expression E.

    The result of the attribute is obtained by evaluating E as a location
    description with the context of the current subprogram, current program
    location, and an empty initial stack.

    The resulting location description L is for a thread lane count sized vector
    of generic type elements. The thread lane count is the value of the
    ``DW_AT_LLVM_lanes`` attribute. Each element holds the conceptual program
    location of the corresponding lane, where the least significant element
    corresponds to the first target architecture specific lane identifier and so
    forth. If the lane was not active when the current subprogram was called,
    its element is an undefined location description.

    ``DW_AT_LLVM_lane_pc`` *allows the compiler to indicate conceptually where
    each lane of a SIMT thread is positioned even when it is in divergent
    control flow that is not active.*

    *Typically, the result is a location description with one composite location
    description with each part being a location description with either one
    undefined location description or one memory location description.*

    If not present, the thread is not being used in a SIMT manner, and the
    thread's current program location is used.

13. ``DW_AT_LLVM_active_lane`` *New*

    For languages that are implemented using a SIMD or SIMT execution model, a
    ``DW_TAG_subprogram``, ``DW_TAG_inlined_subroutine``, or
    ``DW_TAG_entry_point`` debugger information entry may have a
    ``DW_AT_LLVM_active_lane`` attribute whose value is a DWARF expression E.

    The result of the attribute is obtained by evaluating E as a value with the
    context of the current subprogram, current program location, and an empty
    initial stack.

    The DWARF is ill-formed if the resulting value V is not an integral value.

    The resulting V is a bit mask of active lanes for the current program
    location. The N\ :sup:`th` least significant bit of the mask corresponds to
    the N\ :sup:`th` lane. If the bit is 1 the lane is active, otherwise it is
    inactive.

    *Some targets may update the target architecture execution mask for regions
    of code that must execute with different sets of lanes than the current
    active lanes. For example, some code must execute with all lanes made
    temporarily active.* ``DW_AT_LLVM_active_lane`` *allows the compiler to
    provide the means to determine the source language active lanes.*

    If not present and ``DW_AT_LLVM_lanes`` is greater than 1, then the target
    architecture execution mask is used.

14. ``DW_AT_LLVM_vector_size`` *New*

    A ``DW_TAG_base_type`` debugger information entry for a base type T may have
    a ``DW_AT_LLVM_vector_size`` attribute whose value is an integer constant
    that is the vector type size N.

    The representation of a vector base type is as N contiguous elements, each
    one having the representation of a base type T' that is the same as T
    without the ``DW_AT_LLVM_vector_size`` attribute.

    If a ``DW_TAG_base_type`` debugger information entry does not have a
    ``DW_AT_LLVM_vector_size`` attribute, then the base type is not a vector
    type.

    The DWARF is ill-formed if N is not greater than 0.

    .. note::

      LLVM has mention of a non-upstreamed debugger information entry that is
      intended to support vector types. However, that was not for a base type so
      would not be suitable as the type of a stack value entry. But perhaps that
      could be replaced by using this attribute.

15. ``DW_AT_LLVM_augmentation`` *New*

    A ``DW_TAG_compile_unit`` debugger information entry for a compilation unit
    may have a ``DW_AT_LLVM_augmentation`` attribute, whose value is an
    augmentation string.

    *The augmentation string allows producers to indicate that there is
    additional vendor or target specific information in the debugging
    information entries. For example, this might be information about the
    version of vendor specific extensions that are being used.*

    If not present, or if the string is empty, then the compilation unit has no
    augmentation string.

    The format for the augmentation string is:

      | ``[``\ *vendor*\ ``:v``\ *X*\ ``.``\ *Y*\ [\ ``:``\ *options*\ ]\ ``]``\ *

    Where *vendor* is the producer, ``vX.Y`` specifies the major X and minor Y
    version number of the extensions used, and *options* is an optional string
    providing additional information about the extensions. The version number
    must conform to semantic versioning [:ref:`SEMVER <amdgpu-dwarf-SEMVER>`].
    The *options* string must not contain the "\ ``]``\ " character.

    For example:

      ::

        [abc:v0.0][def:v1.2:feature-a=on,feature-b=3]

Program Scope Entities
----------------------

.. _amdgpu-dwarf-language-names:

Unit Entities
~~~~~~~~~~~~~

.. note::

  This augments DWARF Version 5 section 3.1.1 and Table 3.1.

Additional language codes defined for use with the ``DW_AT_language`` attribute
are defined in :ref:`amdgpu-dwarf-language-names-table`.

.. table:: Language Names
   :name: amdgpu-dwarf-language-names-table

   ==================== =============================
   Language Name        Meaning
   ==================== =============================
   ``DW_LANG_LLVM_HIP`` HIP Language.
   ==================== =============================

The HIP language [:ref:`HIP <amdgpu-dwarf-HIP>`] can be supported by extending
the C++ language.

Other Debugger Information
--------------------------

Accelerated Access
~~~~~~~~~~~~~~~~~~

.. _amdgpu-dwarf-lookup-by-name:

Lookup By Name
++++++++++++++

Contents of the Name Index
##########################

.. note::

  The following provides changes to DWARF Version 5 section 6.1.1.1.

  The rule for debugger information entries included in the name index in the
  optional ``.debug_names`` section is extended to also include named
  ``DW_TAG_variable`` debugging information entries with a ``DW_AT_location``
  attribute that includes a ``DW_OP_LLVM_form_aspace_address`` operation.

The name index must contain an entry for each debugging information entry that
defines a named subprogram, label, variable, type, or namespace, subject to the
following rules:

* ``DW_TAG_variable`` debugging information entries with a ``DW_AT_location``
  attribute that includes a ``DW_OP_addr``, ``DW_OP_LLVM_form_aspace_address``,
  or ``DW_OP_form_tls_address`` operation are included; otherwise, they are
  excluded.

Data Representation of the Name Index
#####################################

Section Header
^^^^^^^^^^^^^^

.. note::

  The following provides an addition to DWARF Version 5 section 6.1.1.4.1 item
  14 ``augmentation_string``.

A null-terminated UTF-8 vendor specific augmentation string, which provides
additional information about the contents of this index. If provided, the
recommended format for augmentation string is:

  | ``[``\ *vendor*\ ``:v``\ *X*\ ``.``\ *Y*\ [\ ``:``\ *options*\ ]\ ``]``\ *

Where *vendor* is the producer, ``vX.Y`` specifies the major X and minor Y
version number of the extensions used in the DWARF of the compilation unit, and
*options* is an optional string providing additional information about the
extensions. The version number must conform to semantic versioning [:ref:`SEMVER
<amdgpu-dwarf-SEMVER>`]. The *options* string must not contain the "\ ``]``\ "
character.

For example:

  ::

    [abc:v0.0][def:v1.2:feature-a=on,feature-b=3]

.. note::

  This is different to the definition in DWARF Version 5 but is consistent with
  the other augmentation strings and allows multiple vendor extensions to be
  supported.

.. _amdgpu-dwarf-line-number-information:

Line Number Information
~~~~~~~~~~~~~~~~~~~~~~~

The Line Number Program Header
++++++++++++++++++++++++++++++

Standard Content Descriptions
#############################

.. note::

  This augments DWARF Version 5 section 6.2.4.1.

.. _amdgpu-dwarf-line-number-information-dw-lnct-llvm-source:

1.  ``DW_LNCT_LLVM_source``

    The component is a null-terminated UTF-8 source text string with "\ ``\n``\
    " line endings. This content code is paired with the same forms as
    ``DW_LNCT_path``. It can be used for file name entries.

    The value is an empty null-terminated string if no source is available. If
    the source is available but is an empty file then the value is a
    null-terminated single "\ ``\n``\ ".

    *When the source field is present, consumers can use the embedded source
    instead of attempting to discover the source on disk using the file path
    provided by the* ``DW_LNCT_path`` *field. When the source field is absent,
    consumers can access the file to get the source text.*

    *This is particularly useful for programing languages that support runtime
    compilation and runtime generation of source text. In these cases, the
    source text does not reside in any permanent file. For example, the OpenCL
    language [:ref:`OpenCL <amdgpu-dwarf-OpenCL>`] supports online compilation.*

2.  ``DW_LNCT_LLVM_is_MD5``

    ``DW_LNCT_LLVM_is_MD5`` indicates if the ``DW_LNCT_MD5`` content kind, if
    present, is valid: when 0 it is not valid and when 1 it is valid. If
    ``DW_LNCT_LLVM_is_MD5`` content kind is not present, and ``DW_LNCT_MD5``
    content kind is present, then the MD5 checksum is valid.

    ``DW_LNCT_LLVM_is_MD5`` is always paired with the ``DW_FORM_udata`` form.

    *This allows a compilation unit to have a mixture of files with and without
    MD5 checksums. This can happen when multiple relocatable files are linked
    together.*

.. _amdgpu-dwarf-call-frame-information:

Call Frame Information
~~~~~~~~~~~~~~~~~~~~~~

.. note::

  This section provides changes to existing Call Frame Information and defines
  instructions added by the proposal. Additional support is added for address
  spaces. Register unwind DWARF expressions are generalized to allow any
  location description, including those with composite and implicit location
  descriptions.

  These changes would be incorporated into the DWARF Version 5 section 6.1.

Structure of Call Frame Information
+++++++++++++++++++++++++++++++++++

The register rules are:

*undefined*
  A register that has this rule has no recoverable value in the previous frame.
  (By convention, it is not preserved by a callee.)

*same value*
  This register has not been modified from the previous frame. (By convention,
  it is preserved by the callee, but the callee has not modified it.)

*offset(N)*
  N is a signed byte offset. The previous value of this register is saved at the
  location description computed as if the DWARF operation expression
  ``DW_OP_LLVM_offset N`` is evaluated as a location description with an initial
  stack comprising the location description of the current CFA (see
  :ref:`amdgpu-dwarf-operation-expressions`).

*val_offset(N)*
  N is a signed byte offset. The previous value of this register is the memory
  byte address of the location description computed as if the DWARF operation
  expression ``DW_OP_LLVM_offset N`` is evaluated as a location description with
  an initial stack comprising the location description of the current CFA (see
  :ref:`amdgpu-dwarf-operation-expressions`).

  The DWARF is ill-formed if the CFA location description is not a memory byte
  address location description, or if the register size does not match the size
  of an address in the address space of the current CFA location description.

  *Since the CFA location description is required to be a memory byte address
  location description, the value of val_offset(N) will also be a memory byte
  address location description since it is offsetting the CFA location
  description by N bytes. Furthermore, the value of val_offset(N) will be a
  memory byte address in the same address space as the CFA location
  description.*

  .. note::

    Should DWARF allow the address size to be a different size to the size of
    the register? Requiring them to be the same bit size avoids any issue of
    conversion as the bit contents of the register is simply interpreted as a
    value of the address.

    GDB has a per register hook that allows a target specific conversion on a
    register by register basis. It defaults to truncation of bigger registers,
    and to actually reading bytes from the next register (or reads out of bounds
    for the last register) for smaller registers. There are no GDB tests that
    read a register out of bounds (except an illegal hand written assembly
    test).

*register(R)*
  The previous value of this register is stored in another register numbered R.

  The DWARF is ill-formed if the register sizes do not match.

*expression(E)*
  The previous value of this register is located at the location description
  produced by evaluating the DWARF operation expression E (see
  :ref:`amdgpu-dwarf-operation-expressions`).

  E is evaluated as a location description in the context of the current
  subprogram, current program location, and with an initial stack comprising the
  location description of the current CFA.

*val_expression(E)*
  The previous value of this register is the value produced by evaluating the
  DWARF operation expression E (see :ref:`amdgpu-dwarf-operation-expressions`).

  E is evaluated as a value in the context of the current subprogram, current
  program location, and with an initial stack comprising the location
  description of the current CFA.

  The DWARF is ill-formed if the resulting value type size does not match the
  register size.

  .. note::

    This has limited usefulness as the DWARF expression E can only produce
    values up to the size of the generic type. This is due to not allowing any
    operations that specify a type in a CFI operation expression. This makes it
    unusable for registers that are larger than the generic type. However,
    *expression(E)* can be used to create an implicit location description of
    any size.

*architectural*
  The rule is defined externally to this specification by the augmenter.

A Common Information Entry holds information that is shared among many Frame
Description Entries. There is at least one CIE in every non-empty
``.debug_frame`` section. A CIE contains the following fields, in order:

1.  ``length`` (initial length)

    A constant that gives the number of bytes of the CIE structure, not
    including the length field itself. The size of the length field plus the
    value of length must be an integral multiple of the address size specified
    in the ``address_size`` field.

2.  ``CIE_id`` (4 or 8 bytes, see
    :ref:`amdgpu-dwarf-32-bit-and-64-bit-dwarf-formats`)

    A constant that is used to distinguish CIEs from FDEs.

    In the 32-bit DWARF format, the value of the CIE id in the CIE header is
    0xffffffff; in the 64-bit DWARF format, the value is 0xffffffffffffffff.

3.  ``version`` (ubyte)

    A version number. This number is specific to the call frame information and
    is independent of the DWARF version number.

    The value of the CIE version number is 4.

    .. note::

      Would this be increased to 5 to reflect the changes in the proposal?

4.  ``augmentation`` (sequence of UTF-8 characters)

    A null-terminated UTF-8 string that identifies the augmentation to this CIE
    or to the FDEs that use it. If a reader encounters an augmentation string
    that is unexpected, then only the following fields can be read:

    * CIE: length, CIE_id, version, augmentation
    * FDE: length, CIE_pointer, initial_location, address_range

    If there is no augmentation, this value is a zero byte.

    *The augmentation string allows users to indicate that there is additional
    vendor and target architecture specific information in the CIE or FDE which
    is needed to virtually unwind a stack frame. For example, this might be
    information about dynamically allocated data which needs to be freed on exit
    from the routine.*

    *Because the* ``.debug_frame`` *section is useful independently of any*
    ``.debug_info`` *section, the augmentation string always uses UTF-8
    encoding.*

    The recommended format for the augmentation string is:

      | ``[``\ *vendor*\ ``:v``\ *X*\ ``.``\ *Y*\ [\ ``:``\ *options*\ ]\ ``]``\ *

    Where *vendor* is the producer, ``vX.Y`` specifies the major X and minor Y
    version number of the extensions used, and *options* is an optional string
    providing additional information about the extensions. The version number
    must conform to semantic versioning [:ref:`SEMVER <amdgpu-dwarf-SEMVER>`].
    The *options* string must not contain the "\ ``]``\ " character.

    For example:

      ::

        [abc:v0.0][def:v1.2:feature-a=on,feature-b=3]

5.  ``address_size`` (ubyte)

    The size of a target address in this CIE and any FDEs that use it, in bytes.
    If a compilation unit exists for this frame, its address size must match the
    address size here.

6.  ``segment_selector_size`` (ubyte)

    The size of a segment selector in this CIE and any FDEs that use it, in
    bytes.

7.  ``code_alignment_factor`` (unsigned LEB128)

    A constant that is factored out of all advance location instructions (see
    :ref:`amdgpu-dwarf-row-creation-instructions`). The resulting value is
    ``(operand * code_alignment_factor)``.

8.  ``data_alignment_factor`` (signed LEB128)

    A constant that is factored out of certain offset instructions (see
    :ref:`amdgpu-dwarf-cfa-definition-instructions` and
    :ref:`amdgpu-dwarf-register-rule-instructions`). The resulting value is
    ``(operand * data_alignment_factor)``.

9.  ``return_address_register`` (unsigned LEB128)

    An unsigned LEB128 constant that indicates which column in the rule table
    represents the return address of the subprogram. Note that this column might
    not correspond to an actual machine register.

10. ``initial_instructions`` (array of ubyte)

    A sequence of rules that are interpreted to create the initial setting of
    each column in the table.

    The default rule for all columns before interpretation of the initial
    instructions is the undefined rule. However, an ABI authoring body or a
    compilation system authoring body may specify an alternate default value for
    any or all columns.

11. ``padding`` (array of ubyte)

    Enough ``DW_CFA_nop`` instructions to make the size of this entry match the
    length value above.

An FDE contains the following fields, in order:

1.  ``length`` (initial length)

    A constant that gives the number of bytes of the header and instruction
    stream for this subprogram, not including the length field itself. The size
    of the length field plus the value of length must be an integral multiple of
    the address size.

2.  ``CIE_pointer`` (4 or 8 bytes, see
    :ref:`amdgpu-dwarf-32-bit-and-64-bit-dwarf-formats`)

    A constant offset into the ``.debug_frame`` section that denotes the CIE
    that is associated with this FDE.

3.  ``initial_location`` (segment selector and target address)

    The address of the first location associated with this table entry. If the
    segment_selector_size field of this FDE’s CIE is non-zero, the initial
    location is preceded by a segment selector of the given length.

4.  ``address_range`` (target address)

    The number of bytes of program instructions described by this entry.

5.  ``instructions`` (array of ubyte)

    A sequence of table defining instructions that are described in
    :ref:`amdgpu-dwarf-call-frame-instructions`.

6.  ``padding`` (array of ubyte)

    Enough ``DW_CFA_nop`` instructions to make the size of this entry match the
    length value above.

.. _amdgpu-dwarf-call-frame-instructions:

Call Frame Instructions
+++++++++++++++++++++++

Some call frame instructions have operands that are encoded as DWARF operation
expressions E (see :ref:`amdgpu-dwarf-operation-expressions`). The DWARF
operations that can be used in E have the following restrictions:

* ``DW_OP_addrx``, ``DW_OP_call2``, ``DW_OP_call4``, ``DW_OP_call_ref``,
  ``DW_OP_const_type``, ``DW_OP_constx``, ``DW_OP_convert``,
  ``DW_OP_deref_type``, ``DW_OP_fbreg``, ``DW_OP_implicit_pointer``,
  ``DW_OP_regval_type``, ``DW_OP_reinterpret``, and ``DW_OP_xderef_type``
  operations are not allowed because the call frame information must not depend
  on other debug sections.

* ``DW_OP_push_object_address`` is not allowed because there is no object
  context to provide a value to push.

* ``DW_OP_LLVM_push_lane`` is not allowed because the call frame instructions
  describe the actions for the whole thread, not the lanes independently.

* ``DW_OP_call_frame_cfa`` and ``DW_OP_entry_value`` are not allowed because
  their use would be circular.

* ``DW_OP_LLVM_call_frame_entry_reg`` is not allowed if evaluating E causes a
  circular dependency between ``DW_OP_LLVM_call_frame_entry_reg`` operations.

  *For example, if a register R1 has a* ``DW_CFA_def_cfa_expression``
  *instruction that evaluates a* ``DW_OP_LLVM_call_frame_entry_reg`` *operation
  that specifies register R2, and register R2 has a*
  ``DW_CFA_def_cfa_expression`` *instruction that that evaluates a*
  ``DW_OP_LLVM_call_frame_entry_reg`` *operation that specifies register R1.*

*Call frame instructions to which these restrictions apply include*
``DW_CFA_def_cfa_expression``\ *,* ``DW_CFA_expression``\ *, and*
``DW_CFA_val_expression``\ *.*

.. _amdgpu-dwarf-row-creation-instructions:

Row Creation Instructions
#########################

.. note::

  These instructions are the same as in DWARF Version 5 section 6.4.2.1.

.. _amdgpu-dwarf-cfa-definition-instructions:

CFA Definition Instructions
###########################

1.  ``DW_CFA_def_cfa``

    The ``DW_CFA_def_cfa`` instruction takes two unsigned LEB128 operands
    representing a register number R and a (non-factored) byte displacement B.
    AS is set to the target architecture default address space identifier. The
    required action is to define the current CFA rule to be the result of
    evaluating the DWARF operation expression ``DW_OP_constu AS;
    DW_OP_aspace_bregx R, B`` as a location description.

2.  ``DW_CFA_def_cfa_sf``

    The ``DW_CFA_def_cfa_sf`` instruction takes two operands: an unsigned LEB128
    value representing a register number R and a signed LEB128 factored byte
    displacement B. AS is set to the target architecture default address space
    identifier. The required action is to define the current CFA rule to be the
    result of evaluating the DWARF operation expression ``DW_OP_constu AS;
    DW_OP_aspace_bregx R, B*data_alignment_factor`` as a location description.

    *The action is the same as* ``DW_CFA_def_cfa`` *except that the second
    operand is signed and factored.*

3.  ``DW_CFA_def_aspace_cfa`` *New*

    The ``DW_CFA_def_aspace_cfa`` instruction takes three unsigned LEB128
    operands representing a register number R, a (non-factored) byte
    displacement B, and a target architecture specific address space identifier
    AS. The required action is to define the current CFA rule to be the result
    of evaluating the DWARF operation expression ``DW_OP_constu AS;
    DW_OP_aspace_bregx R, B`` as a location description.

    If AS is not one of the values defined by the target architecture specific
    ``DW_ASPACE_*`` values then the DWARF expression is ill-formed.

4.  ``DW_CFA_def_aspace_cfa_sf`` *New*

    The ``DW_CFA_def_cfa_sf`` instruction takes three operands: an unsigned
    LEB128 value representing a register number R, a signed LEB128 factored byte
    displacement B, and an unsigned LEB128 value representing a target
    architecture specific address space identifier AS. The required action is to
    define the current CFA rule to be the result of evaluating the DWARF
    operation expression ``DW_OP_constu AS; DW_OP_aspace_bregx R,
    B*data_alignment_factor`` as a location description.

    If AS is not one of the values defined by the target architecture specific
    ``DW_ASPACE_*`` values, then the DWARF expression is ill-formed.

    *The action is the same as* ``DW_CFA_aspace_def_cfa`` *except that the
    second operand is signed and factored.*

5.  ``DW_CFA_def_cfa_register``

    The ``DW_CFA_def_cfa_register`` instruction takes a single unsigned LEB128
    operand representing a register number R. The required action is to define
    the current CFA rule to be the result of evaluating the DWARF operation
    expression ``DW_OP_constu AS; DW_OP_aspace_bregx R, B`` as a location
    description. B and AS are the old CFA byte displacement and address space
    respectively.

    If the subprogram has no current CFA rule, or the rule was defined by a
    ``DW_CFA_def_cfa_expression`` instruction, then the DWARF is ill-formed.

6.  ``DW_CFA_def_cfa_offset``

    The ``DW_CFA_def_cfa_offset`` instruction takes a single unsigned LEB128
    operand representing a (non-factored) byte displacement B. The required
    action is to define the current CFA rule to be the result of evaluating the
    DWARF operation expression ``DW_OP_constu AS; DW_OP_aspace_bregx R, B`` as a
    location description. R and AS are the old CFA register number and address
    space respectively.

    If the subprogram has no current CFA rule, or the rule was defined by a
    ``DW_CFA_def_cfa_expression`` instruction, then the DWARF is ill-formed.

7.  ``DW_CFA_def_cfa_offset_sf``

    The ``DW_CFA_def_cfa_offset_sf`` instruction takes a signed LEB128 operand
    representing a factored byte displacement B. The required action is to
    define the current CFA rule to be the result of evaluating the DWARF
    operation expression ``DW_OP_constu AS; DW_OP_aspace_bregx R,
    B*data_alignment_factor`` as a location description. R and AS are the old
    CFA register number and address space respectively.

    If the subprogram has no current CFA rule, or the rule was defined by a
    ``DW_CFA_def_cfa_expression`` instruction, then the DWARF is ill-formed.

    *The action is the same as* ``DW_CFA_def_cfa_offset`` *except that the
    operand is signed and factored.*

8.  ``DW_CFA_def_cfa_expression``

    The ``DW_CFA_def_cfa_expression`` instruction takes a single operand encoded
    as a ``DW_FORM_exprloc`` value representing a DWARF operation expression E.
    The required action is to define the current CFA rule to be the result of
    evaluating E as a location description in the context of the current
    subprogram, current program location, and an empty initial stack.

    *See* :ref:`amdgpu-dwarf-call-frame-instructions` *regarding restrictions on
    the DWARF expression operations that can be used in E.*

    The DWARF is ill-formed if the result of evaluating E is not a memory byte
    address location description.

.. _amdgpu-dwarf-register-rule-instructions:

Register Rule Instructions
##########################

1.  ``DW_CFA_undefined``

    The ``DW_CFA_undefined`` instruction takes a single unsigned LEB128 operand
    that represents a register number R. The required action is to set the rule
    for the register specified by R to ``undefined``.

2.  ``DW_CFA_same_value``

    The ``DW_CFA_same_value`` instruction takes a single unsigned LEB128 operand
    that represents a register number R. The required action is to set the rule
    for the register specified by R to ``same value``.

3.  ``DW_CFA_offset``

    The ``DW_CFA_offset`` instruction takes two operands: a register number R
    (encoded with the opcode) and an unsigned LEB128 constant representing a
    factored displacement B. The required action is to change the rule for the
    register specified by R to be an *offset(B\*data_alignment_factor)* rule.

    .. note::

      Seems this should be named ``DW_CFA_offset_uf`` since the offset is
      unsigned factored.

4.  ``DW_CFA_offset_extended``

    The ``DW_CFA_offset_extended`` instruction takes two unsigned LEB128
    operands representing a register number R and a factored displacement B.
    This instruction is identical to ``DW_CFA_offset`` except for the encoding
    and size of the register operand.

    .. note::

      Seems this should be named ``DW_CFA_offset_extended_uf`` since the
      displacement is unsigned factored.

5.  ``DW_CFA_offset_extended_sf``

    The ``DW_CFA_offset_extended_sf`` instruction takes two operands: an
    unsigned LEB128 value representing a register number R and a signed LEB128
    factored displacement B. This instruction is identical to
    ``DW_CFA_offset_extended`` except that B is signed.

6.  ``DW_CFA_val_offset``

    The ``DW_CFA_val_offset`` instruction takes two unsigned LEB128 operands
    representing a register number R and a factored displacement B. The required
    action is to change the rule for the register indicated by R to be a
    *val_offset(B\*data_alignment_factor)* rule.

    .. note::

      Seems this should be named ``DW_CFA_val_offset_uf`` since the displacement
      is unsigned factored.

    .. note::

      An alternative is to define ``DW_CFA_val_offset`` to implicitly use the
      target architecture default address space, and add another operation that
      specifies the address space.

7.  ``DW_CFA_val_offset_sf``

    The ``DW_CFA_val_offset_sf`` instruction takes two operands: an unsigned
    LEB128 value representing a register number R and a signed LEB128 factored
    displacement B. This instruction is identical to ``DW_CFA_val_offset``
    except that B is signed.

8.  ``DW_CFA_register``

    The ``DW_CFA_register`` instruction takes two unsigned LEB128 operands
    representing register numbers R1 and R2 respectively. The required action is
    to set the rule for the register specified by R1 to be a *register(R2)* rule.

9.  ``DW_CFA_expression``

    The ``DW_CFA_expression`` instruction takes two operands: an unsigned LEB128
    value representing a register number R, and a ``DW_FORM_block`` value
    representing a DWARF operation expression E. The required action is to
    change the rule for the register specified by R to be an *expression(E)*
    rule.

    *That is, E computes the location description where the register value can
    be retrieved.*

    *See* :ref:`amdgpu-dwarf-call-frame-instructions` *regarding restrictions on
    the DWARF expression operations that can be used in E.*

10. ``DW_CFA_val_expression``

    The ``DW_CFA_val_expression`` instruction takes two operands: an unsigned
    LEB128 value representing a register number R, and a ``DW_FORM_block`` value
    representing a DWARF operation expression E. The required action is to
    change the rule for the register specified by R to be a *val_expression(E)*
    rule.

    *That is, E computes the value of register R.*

    *See* :ref:`amdgpu-dwarf-call-frame-instructions` *regarding restrictions on
    the DWARF expression operations that can be used in E.*

    If the result of evaluating E is not a value with a base type size that
    matches the register size, then the DWARF is ill-formed.

11. ``DW_CFA_restore``

    The ``DW_CFA_restore`` instruction takes a single operand (encoded with the
    opcode) that represents a register number R. The required action is to
    change the rule for the register specified by R to the rule assigned it by
    the ``initial_instructions`` in the CIE.

12. ``DW_CFA_restore_extended``

    The ``DW_CFA_restore_extended`` instruction takes a single unsigned LEB128
    operand that represents a register number R. This instruction is identical
    to ``DW_CFA_restore`` except for the encoding and size of the register
    operand.

Row State Instructions
######################

.. note::

  These instructions are the same as in DWARF Version 5 section 6.4.2.4.

Padding Instruction
###################

.. note::

  These instructions are the same as in DWARF Version 5 section 6.4.2.5.

Call Frame Instruction Usage
++++++++++++++++++++++++++++

.. note::

  The same as in DWARF Version 5 section 6.4.3.

.. _amdgpu-dwarf-call-frame-calling-address:

Call Frame Calling Address
++++++++++++++++++++++++++

.. note::

  The same as in DWARF Version 5 section 6.4.4.

Data Representation
-------------------

.. _amdgpu-dwarf-32-bit-and-64-bit-dwarf-formats:

32-Bit and 64-Bit DWARF Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

  This augments DWARF Version 5 section 7.4.

1.  Within the body of the ``.debug_info`` section, certain forms of attribute
    value depend on the choice of DWARF format as follows. For the 32-bit DWARF
    format, the value is a 4-byte unsigned integer; for the 64-bit DWARF format,
    the value is an 8-byte unsigned integer.

    .. table:: ``.debug_info`` section attribute form roles
      :name: amdgpu-dwarf-debug-info-section-attribute-form-roles-table

      ================================== ===================================
      Form                               Role
      ================================== ===================================
      DW_FORM_line_strp                  offset in ``.debug_line_str``
      DW_FORM_ref_addr                   offset in ``.debug_info``
      DW_FORM_sec_offset                 offset in a section other than
                                         ``.debug_info`` or ``.debug_str``
      DW_FORM_strp                       offset in ``.debug_str``
      DW_FORM_strp_sup                   offset in ``.debug_str`` section of
                                         supplementary object file
      DW_OP_call_ref                     offset in ``.debug_info``
      DW_OP_implicit_pointer             offset in ``.debug_info``
      DW_OP_LLVM_aspace_implicit_pointer offset in ``.debug_info``
      ================================== ===================================

Format of Debugging Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Attribute Encodings
+++++++++++++++++++

.. note::

  This augments DWARF Version 5 section 7.5.4 and Table 7.5.

The following table gives the encoding of the additional debugging information
entry attributes.

.. table:: Attribute encodings
   :name: amdgpu-dwarf-attribute-encodings-table

   ================================== ====== ===================================
   Attribute Name                     Value  Classes
   ================================== ====== ===================================
   DW_AT_LLVM_active_lane             0x3e08 exprloc, loclist
   DW_AT_LLVM_augmentation            0x3e09 string
   DW_AT_LLVM_lanes                   0x3e0a constant
   DW_AT_LLVM_lane_pc                 0x3e0b exprloc, loclist
   DW_AT_LLVM_vector_size             0x3e0c constant
   ================================== ====== ===================================

DWARF Expressions
~~~~~~~~~~~~~~~~~

.. note::

  Rename DWARF Version 5 section 7.7 to reflect the unification of location
  descriptions into DWARF expressions.

Operation Expressions
+++++++++++++++++++++

.. note::

  Rename DWARF Version 5 section 7.7.1 and delete section 7.7.2 to reflect the
  unification of location descriptions into DWARF expressions.

  This augments DWARF Version 5 section 7.7.1 and Table 7.9.

The following table gives the encoding of the additional DWARF expression
operations.

.. table:: DWARF Operation Encodings
   :name: amdgpu-dwarf-operation-encodings-table

   ================================== ===== ======== ===============================
   Operation                          Code  Number   Notes
                                            of
                                            Operands
   ================================== ===== ======== ===============================
   DW_OP_LLVM_form_aspace_address     0xe1     0
   DW_OP_LLVM_push_lane               0xe2     0
   DW_OP_LLVM_offset                  0xe3     0
   DW_OP_LLVM_offset_uconst           0xe4     1     ULEB128 byte displacement
   DW_OP_LLVM_bit_offset              0xe5     0
   DW_OP_LLVM_call_frame_entry_reg    0xe6     1     ULEB128 register number
   DW_OP_LLVM_undefined               0xe7     0
   DW_OP_LLVM_aspace_bregx            0xe8     2     ULEB128 register number,
                                                     ULEB128 byte displacement
   DW_OP_LLVM_aspace_implicit_pointer 0xe9     2     4- or 8-byte offset of DIE,
                                                     SLEB128 byte displacement
   DW_OP_LLVM_piece_end               0xea     0
   DW_OP_LLVM_extend                  0xeb     2     ULEB128 bit size,
                                                     ULEB128 count
   DW_OP_LLVM_select_bit_piece        0xec     2     ULEB128 bit size,
                                                     ULEB128 count
   ================================== ===== ======== ===============================

Location List Expressions
+++++++++++++++++++++++++

.. note::

  Rename DWARF Version 5 section 7.7.3 to reflect that location lists are a kind
  of DWARF expression.

Source Languages
~~~~~~~~~~~~~~~~

.. note::

  This augments DWARF Version 5 section 7.12 and Table 7.17.

The following table gives the encoding of the additional DWARF languages.

.. table:: Language encodings
   :name: amdgpu-dwarf-language-encodings-table

   ==================== ====== ===================
   Language Name        Value  Default Lower Bound
   ==================== ====== ===================
   ``DW_LANG_LLVM_HIP`` 0x8100 0
   ==================== ====== ===================

Address Class and Address Space Encodings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

  This replaces DWARF Version 5 section 7.13.

The encodings of the constants used for the currently defined address classes
are given in :ref:`amdgpu-dwarf-address-class-encodings-table`.

.. table:: Address class encodings
   :name: amdgpu-dwarf-address-class-encodings-table

   ========================== ======
   Address Class Name         Value
   ========================== ======
   ``DW_ADDR_none``           0x0000
   ``DW_ADDR_LLVM_global``    0x0001
   ``DW_ADDR_LLVM_constant``  0x0002
   ``DW_ADDR_LLVM_group``     0x0003
   ``DW_ADDR_LLVM_private``   0x0004
   ``DW_ADDR_LLVM_lo_user``   0x8000
   ``DW_ADDR_LLVM_hi_user``   0xffff
   ========================== ======

Line Number Information
~~~~~~~~~~~~~~~~~~~~~~~

.. note::

  This augments DWARF Version 5 section 7.22 and Table 7.27.

The following table gives the encoding of the additional line number header
entry formats.

.. table:: Line number header entry format encodings
  :name: amdgpu-dwarf-line-number-header-entry-format-encodings-table

  ====================================  ====================
  Line number header entry format name  Value
  ====================================  ====================
  ``DW_LNCT_LLVM_source``               0x2001
  ``DW_LNCT_LLVM_is_MD5``               0x2002
  ====================================  ====================

Call Frame Information
~~~~~~~~~~~~~~~~~~~~~~

.. note::

  This augments DWARF Version 5 section 7.24 and Table 7.29.

The following table gives the encoding of the additional call frame information
instructions.

.. table:: Call frame instruction encodings
   :name: amdgpu-dwarf-call-frame-instruction-encodings-table

   ======================== ====== ====== ================ ================ ================
   Instruction              High 2 Low 6  Operand 1        Operand 2        Operand 3
                            Bits   Bits
   ======================== ====== ====== ================ ================ ================
   DW_CFA_def_aspace_cfa    0      0x30   ULEB128 register ULEB128 offset   ULEB128 address space
   DW_CFA_def_aspace_cfa_sf 0      0x31   ULEB128 register SLEB128 offset   ULEB128 address space
   ======================== ====== ====== ================ ================ ================

Attributes by Tag Value (Informative)
-------------------------------------

.. note::

  This augments DWARF Version 5 Appendix A and Table A.1.

The following table provides the additional attributes that are applicable to
debugger information entries.

.. table:: Attributes by tag value
   :name: amdgpu-dwarf-attributes-by-tag-value-table

   ============================= =============================
   Tag Name                      Applicable Attributes
   ============================= =============================
   ``DW_TAG_base_type``          * ``DW_AT_LLVM_vector_size``
   ``DW_TAG_compile_unit``       * ``DW_AT_LLVM_augmentation``
   ``DW_TAG_entry_point``        * ``DW_AT_LLVM_active_lane``
                                 * ``DW_AT_LLVM_lane_pc``
                                 * ``DW_AT_LLVM_lanes``
   ``DW_TAG_inlined_subroutine`` * ``DW_AT_LLVM_active_lane``
                                 * ``DW_AT_LLVM_lane_pc``
                                 * ``DW_AT_LLVM_lanes``
   ``DW_TAG_subprogram``         * ``DW_AT_LLVM_active_lane``
                                 * ``DW_AT_LLVM_lane_pc``
                                 * ``DW_AT_LLVM_lanes``
   ============================= =============================

.. _amdgpu-dwarf-examples:

Examples
========

The AMD GPU specific usage of the features in the proposal, including examples,
is available at :ref:`amdgpu-dwarf-debug-information`.

.. _amdgpu-dwarf-references:

References
==========

    .. _amdgpu-dwarf-AMD:

1.  [AMD] `Advanced Micro Devices <https://www.amd.com/>`__

    .. _amdgpu-dwarf-AMD-ROCm:

2.  [AMD-ROCm] `AMD ROCm Platform <https://rocm-documentation.readthedocs.io>`__

    .. _amdgpu-dwarf-AMD-ROCgdb:

3.  [AMD-ROCgdb] `AMD ROCm Debugger (ROCgdb) <https://github.com/ROCm-Developer-Tools/ROCgdb>`__

    .. _amdgpu-dwarf-AMDGPU-LLVM:

4.  [AMDGPU-LLVM] `User Guide for AMDGPU LLVM Backend <https://llvm.org/docs/AMDGPUUsage.html>`__

    .. _amdgpu-dwarf-CUDA:

5.  [CUDA] `Nvidia CUDA Language <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>`__

    .. _amdgpu-dwarf-DWARF:

6.  [DWARF] `DWARF Debugging Information Format <http://dwarfstd.org/>`__

    .. _amdgpu-dwarf-ELF:

7.  [ELF] `Executable and Linkable Format (ELF) <http://www.sco.com/developers/gabi/>`__

    .. _amdgpu-dwarf-GCC:

8.  [GCC] `GCC: The GNU Compiler Collection <https://www.gnu.org/software/gcc/>`__

    .. _amdgpu-dwarf-GDB:

9.  [GDB] `GDB: The GNU Project Debugger <https://www.gnu.org/software/gdb/>`__

    .. _amdgpu-dwarf-HIP:

10. [HIP] `HIP Programming Guide <https://rocm-documentation.readthedocs.io/en/latest/Programming_Guides/Programming-Guides.html#hip-programing-guide>`__

    .. _amdgpu-dwarf-HSA:

11. [HSA] `Heterogeneous System Architecture (HSA) Foundation <http://www.hsafoundation.com/>`__

    .. _amdgpu-dwarf-LLVM:

12. [LLVM] `The LLVM Compiler Infrastructure <https://llvm.org/>`__

    .. _amdgpu-dwarf-OpenCL:

13. [OpenCL] `The OpenCL Specification Version 2.0 <http://www.khronos.org/registry/cl/specs/opencl-2.0.pdf>`__

    .. _amdgpu-dwarf-Perforce-TotalView:

14. [Perforce-TotalView] `Perforce TotalView HPC Debugging Software <https://totalview.io/products/totalview>`__

    .. _amdgpu-dwarf-SEMVER:

15. [SEMVER] `Semantic Versioning <https://semver.org/>`__
