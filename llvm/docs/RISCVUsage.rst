=============================
User Guide for RISC-V Target
=============================

.. contents::
   :local:

Introduction
============

The RISC-V target provides code generation for processors implementing
supported variations of the RISC-V specification.  It lives in the
``llvm/lib/Target/RISCV`` directory.

Base ISAs
=========

The specification defines four base instruction sets: RV32I, RV32E, RV64I,
and RV128I. Currently, LLVM fully supports RV32I, and RV64I.  RV32E is
supported by the assembly-based tools only.  RV128I is not supported.

To specify the target triple:

  .. table:: RISC-V Architectures

     ============ ==============================================================
     Architecture Description
     ============ ==============================================================
     ``riscv32``   RISC-V with XLEN=32 (i.e. RV32I or RV32E)
     ``riscv64``   RISC-V with XLEN=64 (i.e. RV64I)
     ============ ==============================================================

To select an E variant ISA (e.g. RV32E instead of RV32I), use the base
architecture string (e.g. ``riscv32``) with the extension ``e``.

.. _riscv-extensions:

Extensions
==========

The following table provides a status summary for extensions which have been
ratified and thus have finalized specifications.  When relevant, detailed notes
on support follow.

  .. table:: Ratified Extensions by Status

     ===============  =========================================================
     Extension        Status
     ===============  =========================================================
     ``A``            Supported
     ``C``            Supported
     ``D``            Supported
     ``F``            Supported
     ``H``            Assembly Support
     ``M``            Supported
     ``Svinval``      Assembly Support
     ``Svnapot``      Assembly Support
     ``Svpbmt``       Supported
     ``V``            Supported
     ``Zba``          Supported
     ``Zbb``          Supported
     ``Zbc``          Supported
     ``Zbkb``         Supported (`See note <#riscv-scalar-crypto-note1>`__)
     ``Zbkc``         Supported
     ``Zbkx``         Supported (`See note <#riscv-scalar-crypto-note1>`__)
     ``Zbs``          Supported
     ``Zdinx``        Assembly Support
     ``Zfh``          Supported
     ``Zfhmin``       Supported
     ``Zfinx``        Assembly Support
     ``Zhinx``        Assembly Support
     ``Zhinxmin``     Assembly Support
     ``Zicbom``       Assembly Support
     ``Zicbop``       Assembly Support
     ``Zicboz``       Assembly Support
     ``Zicsr``        (`See Note <#riscv-i2p1-note>`__)
     ``Zifencei``     (`See Note <#riscv-i2p1-note>`__)
     ``Zihintpause``  Assembly Support
     ``Zkn``          Supported
     ``Zknd``         Supported (`See note <#riscv-scalar-crypto-note2>`__)
     ``Zkne``         Supported (`See note <#riscv-scalar-crypto-note2>`__)
     ``Zknh``         Supported (`See note <#riscv-scalar-crypto-note2>`__)
     ``Zksed``        Supported (`See note <#riscv-scalar-crypto-note2>`__)
     ``Zksh``         Supported (`See note <#riscv-scalar-crypto-note2>`__)
     ``Zk``           Supported
     ``Zkr``          Supported
     ``Zks``          Supported
     ``Zkt``          Supported
     ``Zmmul``        Supported
     ``Zve32x``       (`Partially <#riscv-vlen-32-note>`__) Supported
     ``Zve32f``       (`Partially <#riscv-vlen-32-note>`__) Supported
     ``Zve64x``       Supported
     ``Zve64f``       Supported
     ``Zve64d``       Supported
     ``Zvl32b``       (`Partially <#riscv-vlen-32-note>`__) Supported
     ``Zvl64b``       Supported
     ``Zvl128b``      Supported
     ``Zvl256b``      Supported
     ``Zvl512b``      Supported
     ``Zvl1024b``     Supported
     ``Zvl2048b``     Supported
     ``Zvl4096b``     Supported
     ``Zvl8192b``     Supported
     ``Zvl16384b``    Supported
     ``Zvl32768b``    Supported
     ``Zvl65536b``    Supported
     ===============  =========================================================

Assembly Support
  LLVM supports the associated instructions in assembly.  All assembly related tools (e.g. assembler, disassembler, llvm-objdump, etc..) are supported.  Compiler and linker will accept extension names, and linked binaries will contain appropriate ELF flags and attributes to reflect use of named extension.

Supported
  Fully supported by the compiler.  This includes everything in Assembly Support, along with - if relevant - C language intrinsics for the instructions and pattern matching by the compiler to recognize idiomatic patterns which can be lowered to the associated instructions.

.. _riscv-scalar-crypto-note1:

``Zbkb``, ``Zbkx``
  Pattern matching support for these instructions is incomplete.

.. _riscv-scalar-crypto-note2:

``Zknd``, ``Zkne``, ``Zknh``, ``Zksed``, ``Zksh``
  No pattern matching exists.  As a result, these instructions can only be used from assembler or via intrinsic calls.

.. _riscv-vlen-32-note:

``Zve32x``, ``Zve32f``, ``Zvl32b``
  LLVM currently assumes a minimum VLEN (vector register width) of 64 bits during compilation, and as a result ``Zve32x`` and ``Zve32f`` are supported only for VLEN>=64.  Assembly support doesn't have this restriction.

.. _riscv-i2p1-note:

``zicsr``, ``zifencei``
  Between versions 2.0 and 2.1 of the base I specification, a backwards incompatible change was made to remove selected instructions and CSRs from the base ISA.  These instructions were grouped into a set of new extensions, but were no longer required by the base ISA.  This change is described in "Preface to Document Version 20190608-Base-Ratified" from the specification document.  LLVM currently implements version 2.0 of the base specification.  Thus, instructions from these extensions are accepted as part of the base ISA, but attempts to explicitly enable the extensions will error.

Experimental Extensions
=======================

LLVM supports (to various degrees) a number of experimental extensions.  All experimental extensions have ``experimental-`` as a prefix.  There is explicitly no compatibility promised between versions of the toolchain, and regular users are strongly advised *not* to make use of experimental extensions before they reach ratification.

The primary goal of experimental support is to assist in the process of ratification by providing an existence proof of an implementation, and simplifying efforts to validate the value of a proposed extension against large code bases.  Experimental extensions are expected to either transition to ratified status, or be eventually removed.  The decision on whether to accept an experimental extension is currently done on an entirely case by case basis; if you want to propose one, attending the bi-weekly RISC-V sync-up call is strongly advised.

``experimental-zawrs``
  LLVM implements the `1.0-rc3 draft specification <https://github.com/riscv/riscv-zawrs/releases/download/V1.0-rc3/Zawrs.pdf>`_.  Note that have been backwards incompatible changes made between release candidates for the 1.0 draft.

``experimental-zca``
  LLVM implements the `1.0.1 draft specification <https://github.com/riscv/riscv-code-size-reduction/releases/tag/v1.0.1>`_.

``experimental-zcb``
  LLVM implements the `1.0.1 draft specification <https://github.com/riscv/riscv-code-size-reduction/releases/tag/v1.0.1>`_.

``experimental-zcd``
  LLVM implements the `1.0.1 draft specification <https://github.com/riscv/riscv-code-size-reduction/releases/tag/v1.0.1>`_.

``experimental-zcf``
  LLVM implements the `1.0.1 draft specification <https://github.com/riscv/riscv-code-size-reduction/releases/tag/v1.0.1>`_.

``experimental-zihintntl``
  LLVM implements the `0.2 draft specification <https://github.com/riscv/riscv-isa-manual/releases/tag/draft-20220831-bf5a151>`_.

``experimental-ztso``
  LLVM implements the `v0.1 proposed specification <https://github.com/riscv/riscv-isa-manual/releases/download/draft-20220723-10eea63/riscv-spec.pdf>`_ (see Chapter 25).  The mapping from the C/C++ memory model to Ztso has not yet been ratified in any standards document.  There are multiple possible mappings, and they are *not* mutually ABI compatible.  The mapping LLVM implements is ABI compatible with the default WMO mapping.  This mapping may change and there is *explicitly* no ABI stability offered while the extension remains in experimental status.  User beware.

``experimental-zvfh``
  LLVM implements `this draft text <https://github.com/riscv/riscv-v-spec/pull/780>`_.

``experimental-zfa``
  LLVM implements a subset of `0.1 draft specification <https://github.com/riscv/riscv-isa-manual/releases/download/draft-20221119-5234c63/riscv-spec.pdf>`_ (see Chapter 25). Load-immediate instructions (fli.s/fli.d/fli.h) haven't been implemented yet.
  
To use an experimental extension from `clang`, you must add `-menable-experimental-extensions` to the command line, and specify the exact version of the experimental extension you are using.  To use an experimental extension with LLVM's internal developer tools (e.g. `llc`, `llvm-objdump`, `llvm-mc`), you must prefix the extension name with `experimental-`.  Note that you don't need to specify the version with internal tools, and shouldn't include the `experimental-` prefix with `clang`.

Vendor Extensions
=================

Vendor extensions are extensions which are not standardized by RISC-V International, and are instead defined by a hardware vendor.  The term vendor extension roughly parallels the definition of a `non-standard` extension from Section 1.3 of the Volume I: RISC-V Unprivileged ISA specification.  In particular, we expect to eventually accept both `custom` extensions and `non-conforming` extensions.

Inclusion of a vendor extension will be considered on a case by case basis.  All proposals should be brought to the bi-weekly RISCV sync calls for discussion.  For a general idea of the factors likely to be considered, please see the `Clang documentation <https://clang.llvm.org/get_involved.html>`_.

It is our intention to follow the naming conventions described in `riscv-non-isa/riscv-toolchain-conventions <https://github.com/riscv-non-isa/riscv-toolchain-conventions#conventions-for-vendor-extensions>`_.  Exceptions to this naming will need to be strongly motivated.

The current vendor extensions supported are:

``XTHeadBa``
  LLVM implements `the THeadBa (address-generation) vendor-defined instructions specified in <https://github.com/T-head-Semi/thead-extension-spec/releases/download/2.2.2/xthead-2023-01-30-2.2.2.pdf>`_  by T-HEAD of Alibaba.  Instructions are prefixed with `th.` as described in the specification.

``XTHeadBb``
  LLVM implements `the THeadBb (basic bit-manipulation) vendor-defined instructions specified in <https://github.com/T-head-Semi/thead-extension-spec/releases/download/2.2.2/xthead-2023-01-30-2.2.2.pdf>`_  by T-HEAD of Alibaba.  Instructions are prefixed with `th.` as described in the specification.

``XTHeadBs``
  LLVM implements `the THeadBs (single-bit operations) vendor-defined instructions specified in <https://github.com/T-head-Semi/thead-extension-spec/releases/download/2.2.2/xthead-2023-01-30-2.2.2.pdf>`_  by T-HEAD of Alibaba.  Instructions are prefixed with `th.` as described in the specification.

``XTheadMac``
  LLVM implements `the XTheadMac (multiply-accumulate instructions) vendor-defined instructions specified in <https://github.com/T-head-Semi/thead-extension-spec/releases/download/2.2.2/xthead-2023-01-30-2.2.2.pdf>`_  by T-HEAD of Alibaba.  Instructions are prefixed with `th.` as described in the specification.

``XTHeadVdot``
  LLVM implements `version 1.0.0 of the THeadV-family custom instructions specification <https://github.com/T-head-Semi/thead-extension-spec/releases/download/2.2.0/xthead-2022-12-04-2.2.0.pdf>`_ by T-HEAD of Alibaba.  All instructions are prefixed with `th.` as described in the specification, and the riscv-toolchain-convention document linked above.

``XVentanaCondOps``
  LLVM implements `version 1.0.0 of the VTx-family custom instructions specification <https://github.com/ventanamicro/ventana-custom-extensions/releases/download/v1.0.0/ventana-custom-extensions-v1.0.0.pdf>`_ by Ventana Micro Systems.  All instructions are prefixed with `vt.` as described in the specification, and the riscv-toolchain-convention document linked above.  These instructions are only available for riscv64 at this time.

Specification Documents
=======================
For ratified specifications, please refer to the `official RISC-V International
page <https://riscv.org/technical/specifications/>`_.  Make sure to check the
`wiki for not yet integrated extensions
<https://wiki.riscv.org/display/HOME/Recently+Ratified+Extensions>`_.

