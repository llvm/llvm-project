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

Specification Documents
=======================

There have been a number of revisions to the RISC-V specifications. LLVM aims
to implement the most recent ratified version of the standard RISC-V base ISAs
and ISA extensions with pragmatic variances. The most recent specification can
be found at: https://github.com/riscv/riscv-isa-manual/releases/.

`The official RISC-V International specification page
<https://riscv.org/technical/specifications/>`__. is also worth checking, but
tends to significantly lag the specifications linked above. Make sure to check
the `wiki for not yet integrated extensions
<https://wiki.riscv.org/display/HOME/Recently+Ratified+Extensions>`__ and note
that in addition, we sometimes carry support for extensions that have not yet
been ratified (these will be marked as experimental - see below) and support
various vendor-specific extensions (see below).

The current known variances from the specification are:

* Unconditionally allowing instructions from zifencei, zicsr, zicntr, and
  zihpm without gating them on the extensions being enabled.  Previous
  revisions of the specification included these instructions in the base
  ISA, and we preserve this behavior to avoid breaking existing code.  If
  a future revision of the specification reuses these opcodes for other
  extensions, we may need to reevaluate this choice, and thus recommend
  users migrate build systems so as not to rely on this.
* Allowing CSRs to be named without gating on specific extensions.  This
  applies to all CSR names, not just those in zicsr, zicntr, and zihpm.
* The ordering of ``z*``, ``s*``, and ``x*`` prefixed extension names is not
  enforced in user-specified ISA naming strings (e.g. ``-march``).

We are actively deciding not to support multiple specification revisions
at this time. We acknowledge a likely future need, but actively defer the
decisions making around handling this until we have a concrete example of
real hardware having shipped and an incompatible change to the
specification made afterwards.

Base ISAs
=========

The specification defines five base instruction sets: RV32I, RV32E, RV64I,
RV64E, and RV128I. Currently, LLVM fully supports RV32I, and RV64I.  RV32E and
RV64E are supported by the assembly-based tools only.  RV128I is not supported.

To specify the target triple:

  .. table:: RISC-V Architectures

     ============ ==============================================================
     Architecture Description
     ============ ==============================================================
     ``riscv32``   RISC-V with XLEN=32 (i.e. RV32I or RV32E)
     ``riscv64``   RISC-V with XLEN=64 (i.e. RV64I or RV64E)
     ============ ==============================================================

To select an E variant ISA (e.g. RV32E instead of RV32I), use the base
architecture string (e.g. ``riscv32``) with the extension ``e``.

Profiles
========

Supported profile names can be passed using ``-march`` instead of a standard
ISA naming string. Currently supported profiles:

* ``rvi20u32``
* ``rvi20u64``
* ``rva20u64``
* ``rva20s64``
* ``rva22u64``
* ``rva22s64``
* ``rva23u64``
* ``rva23s64``
* ``rvb23u64``
* ``rvb23s64``

Note that you can also append additional extension names to be enabled, e.g.
``rva20u64_zicond`` will enable the ``zicond`` extension in addition to those
in the ``rva20u64`` profile.

Profiles that are not yet ratified cannot be used unless
``-menable-experimental-extensions`` (or equivalent for other tools) is
specified. This applies to the following profiles:

* ``rvm23u32``

.. _riscv-extensions:

Extensions
==========

The following table provides a status summary for extensions which have been
ratified and thus have finalized specifications.  When relevant, detailed notes
on support follow.

  .. table:: Ratified Extensions by Status

     ================  =================================================================
     Extension         Status
     ================  =================================================================
     ``A``             Supported
     ``B``             Supported
     ``C``             Supported
     ``D``             Supported
     ``F``             Supported
     ``E``             Supported (`See note <#riscv-rve-note>`__)
     ``H``             Assembly Support
     ``M``             Supported
     ``Sha``           Supported
     ``Shcounterenw``  Assembly Support (`See note <#riscv-profiles-extensions-note>`__)
     ``Shgatpa``       Assembly Support (`See note <#riscv-profiles-extensions-note>`__)
     ``Shtvala``       Assembly Support (`See note <#riscv-profiles-extensions-note>`__)
     ``Shvsatpa``      Assembly Support (`See note <#riscv-profiles-extensions-note>`__)
     ``Shvstvala``     Assembly Support (`See note <#riscv-profiles-extensions-note>`__)
     ``Shvstvecd``     Assembly Support (`See note <#riscv-profiles-extensions-note>`__)
     ``Smaia``         Supported
     ``Smcdeleg``      Supported
     ``Smcsrind``      Supported
     ``Smdbltrp``      Supported
     ``Smepmp``        Supported
     ``Smmpm``         Supported
     ``Smnpm``         Supported
     ``Smrnmi``        Assembly Support
     ``Smstateen``     Assembly Support
     ``Ssaia``         Supported
     ``Ssccfg``        Supported
     ``Ssccptr``       Assembly Support (`See note <#riscv-profiles-extensions-note>`__)
     ``Sscofpmf``      Assembly Support
     ``Sscounterenw``  Assembly Support (`See note <#riscv-profiles-extensions-note>`__)
     ``Sscsrind``      Supported
     ``Ssdbltrp``      Supported
     ``Ssnpm``         Supported
     ``Sspm``          Supported
     ``Ssqosid``       Assembly Support
     ``Ssstateen``     Assembly Support (`See note <#riscv-profiles-extensions-note>`__)
     ``Ssstrict``      Assembly Support (`See note <#riscv-profiles-extensions-note>`__)
     ``Sstc``          Assembly Support
     ``Sstvala``       Assembly Support (`See note <#riscv-profiles-extensions-note>`__)
     ``Sstvecd``       Assembly Support (`See note <#riscv-profiles-extensions-note>`__)
     ``Ssu64xl``       Assembly Support (`See note <#riscv-profiles-extensions-note>`__)
     ``Supm``          Supported
     ``Svade``         Assembly Support (`See note <#riscv-profiles-extensions-note>`__)
     ``Svadu``         Assembly Support
     ``Svbare``        Assembly Support (`See note <#riscv-profiles-extensions-note>`__)
     ``Svinval``       Assembly Support
     ``Svnapot``       Assembly Support
     ``Svpbmt``        Supported
     ``Svvptc``        Supported
     ``V``             Supported
     ``Za128rs``       Supported (`See note <#riscv-profiles-extensions-note>`__)
     ``Za64rs``        Supported (`See note <#riscv-profiles-extensions-note>`__)
     ``Zaamo``         Assembly Support
     ``Zabha``         Supported
     ``Zacas``         Supported (`See note <#riscv-zacas-note>`__)
     ``Zalrsc``        Assembly Support
     ``Zama16b``       Supported (`See note <#riscv-profiles-extensions-note>`__)
     ``Zawrs``         Assembly Support
     ``Zba``           Supported
     ``Zbb``           Supported
     ``Zbc``           Supported
     ``Zbkb``          Supported (`See note <#riscv-scalar-crypto-note1>`__)
     ``Zbkc``          Supported
     ``Zbkx``          Supported (`See note <#riscv-scalar-crypto-note1>`__)
     ``Zbs``           Supported
     ``Zca``           Supported
     ``Zcb``           Supported
     ``Zcd``           Supported
     ``Zcf``           Supported
     ``Zcmop``         Supported
     ``Zcmp``          Supported
     ``Zcmt``          Assembly Support
     ``Zdinx``         Supported
     ``Zfa``           Supported
     ``Zfbfmin``       Supported
     ``Zfh``           Supported
     ``Zfhmin``        Supported
     ``Zfinx``         Supported
     ``Zhinx``         Supported
     ``Zhinxmin``      Supported
     ``Zic64b``        Supported (`See note <#riscv-profiles-extensions-note>`__)
     ``Zicbom``        Assembly Support
     ``Zicbop``        Supported
     ``Zicboz``        Assembly Support
     ``Ziccamoa``      Supported (`See note <#riscv-profiles-extensions-note>`__)
     ``Ziccif``        Supported (`See note <#riscv-profiles-extensions-note>`__)
     ``Zicclsm``       Supported (`See note <#riscv-profiles-extensions-note>`__)
     ``Ziccrse``       Supported (`See note <#riscv-profiles-extensions-note>`__)
     ``Zicntr``        (`See Note <#riscv-i2p1-note>`__)
     ``Zicond``        Supported
     ``Zicsr``         (`See Note <#riscv-i2p1-note>`__)
     ``Zifencei``      (`See Note <#riscv-i2p1-note>`__)
     ``Zihintntl``     Supported
     ``Zihintpause``   Assembly Support
     ``Zihpm``         (`See Note <#riscv-i2p1-note>`__)
     ``Zimop``         Supported
     ``Zkn``           Supported
     ``Zknd``          Supported (`See note <#riscv-scalar-crypto-note2>`__)
     ``Zkne``          Supported (`See note <#riscv-scalar-crypto-note2>`__)
     ``Zknh``          Supported (`See note <#riscv-scalar-crypto-note2>`__)
     ``Zksed``         Supported (`See note <#riscv-scalar-crypto-note2>`__)
     ``Zksh``          Supported (`See note <#riscv-scalar-crypto-note2>`__)
     ``Zk``            Supported
     ``Zkr``           Supported
     ``Zks``           Supported
     ``Zkt``           Supported
     ``Zmmul``         Supported
     ``Ztso``          Supported
     ``Zvbb``          Supported
     ``Zvbc``          Supported (`See note <#riscv-vector-crypto-note>`__)
     ``Zve32x``        (`Partially <#riscv-vlen-32-note>`__) Supported
     ``Zve32f``        (`Partially <#riscv-vlen-32-note>`__) Supported
     ``Zve64x``        Supported
     ``Zve64f``        Supported
     ``Zve64d``        Supported
     ``Zvfbfmin``      Supported
     ``Zvfbfwma``      Supported
     ``Zvfh``          Supported
     ``Zvfhmin``       Supported
     ``Zvkb``          Supported
     ``Zvkg``          Supported (`See note <#riscv-vector-crypto-note>`__)
     ``Zvkn``          Supported (`See note <#riscv-vector-crypto-note>`__)
     ``Zvknc``         Supported (`See note <#riscv-vector-crypto-note>`__)
     ``Zvkned``        Supported (`See note <#riscv-vector-crypto-note>`__)
     ``Zvkng``         Supported (`See note <#riscv-vector-crypto-note>`__)
     ``Zvknha``        Supported (`See note <#riscv-vector-crypto-note>`__)
     ``Zvknhb``        Supported (`See note <#riscv-vector-crypto-note>`__)
     ``Zvks``          Supported (`See note <#riscv-vector-crypto-note>`__)
     ``Zvksc``         Supported (`See note <#riscv-vector-crypto-note>`__)
     ``Zvksed``        Supported (`See note <#riscv-vector-crypto-note>`__)
     ``Zvksg``         Supported (`See note <#riscv-vector-crypto-note>`__)
     ``Zvksh``         Supported (`See note <#riscv-vector-crypto-note>`__)
     ``Zvkt``          Supported
     ``Zvl32b``        (`Partially <#riscv-vlen-32-note>`__) Supported
     ``Zvl64b``        Supported
     ``Zvl128b``       Supported
     ``Zvl256b``       Supported
     ``Zvl512b``       Supported
     ``Zvl1024b``      Supported
     ``Zvl2048b``      Supported
     ``Zvl4096b``      Supported
     ``Zvl8192b``      Supported
     ``Zvl16384b``     Supported
     ``Zvl32768b``     Supported
     ``Zvl65536b``     Supported
     ================  =================================================================

Assembly Support
  LLVM supports the associated instructions in assembly.  All assembly related tools (e.g. assembler, disassembler, llvm-objdump, etc..) are supported.  Compiler and linker will accept extension names, and linked binaries will contain appropriate ELF flags and attributes to reflect use of named extension.

Supported
  Fully supported by the compiler.  This includes everything in Assembly Support, along with - if relevant - C language intrinsics for the instructions and pattern matching by the compiler to recognize idiomatic patterns which can be lowered to the associated instructions.

.. _riscv-rve-note:

``E``
  Support of RV32E/RV64E and ilp32e/lp64e ABIs are experimental. To be compatible with the implementation of ilp32e in GCC, we don't use aligned registers to pass variadic arguments. Furthermore, we set the stack alignment to 4 bytes for types with length of 2*XLEN.

.. _riscv-scalar-crypto-note1:

``Zbkb``, ``Zbkx``
  Pattern matching support for these instructions is incomplete.

.. _riscv-scalar-crypto-note2:

``Zknd``, ``Zkne``, ``Zknh``, ``Zksed``, ``Zksh``
  No pattern matching exists.  As a result, these instructions can only be used from assembler or via intrinsic calls.

.. _riscv-vector-crypto-note:

``Zvbc``, ``Zvkg``, ``Zvkn``, ``Zvknc``, ``Zvkned``, ``Zvkng``, ``Zvknha``, ``Zvknhb``, ``Zvks``, ``Zvks``, ``Zvks``, ``Zvksc``, ``Zvksed``, ``Zvksg``, ``Zvksh``.
  No pattern matching exists. As a result, these instructions can only be used from assembler or via intrinsic calls.

.. _riscv-vlen-32-note:

``Zve32x``, ``Zve32f``, ``Zvl32b``
  LLVM currently assumes a minimum VLEN (vector register width) of 64 bits during compilation, and as a result ``Zve32x`` and ``Zve32f`` are supported only for VLEN>=64.  Assembly support doesn't have this restriction.

.. _riscv-i2p1-note:

``Zicntr``, ``Zicsr``, ``Zifencei``, ``Zihpm``
  Between versions 2.0 and 2.1 of the base I specification, a backwards incompatible change was made to remove selected instructions and CSRs from the base ISA.  These instructions were grouped into a set of new extensions, but were no longer required by the base ISA.  This change is partially described in "Preface to Document Version 20190608-Base-Ratified" from the specification document (the ``zicntr`` and ``zihpm`` bits are not mentioned).  LLVM currently implements version 2.1 of the base specification. To maintain compatibility, instructions from these extensions are accepted without being in the ``-march`` string.  LLVM also allows the explicit specification of the extensions in an ``-march`` string.

.. _riscv-profiles-extensions-note:

``Za128rs``, ``Za64rs``, ``Zama16b``, ``Zic64b``, ``Ziccamoa``, ``Ziccif``, ``Zicclsm``, ``Ziccrse``, ``Shcounterenvw``, ``Shgatpa``, ``Shtvala``, ``Shvsatpa``, ``Shvstvala``, ``Shvstvecd``, ``Ssccptr``, ``Sscounterenw``, ``Ssstateen``, ``Ssstrict``, ``Sstvala``, ``Sstvecd``, ``Ssu64xl``, ``Svade``, ``Svbare``
  These extensions are defined as part of the `RISC-V Profiles specification <https://github.com/riscv/riscv-profiles/releases/tag/v1.0>`__.  They do not introduce any new features themselves, but instead describe existing hardware features.

.. _riscv-zacas-note:

``Zacas``
  The compiler will not generate amocas.d on RV32 or amocas.q on RV64 due to ABI compatibilty. These can only be used in the assembler.

Atomics ABIs
============

At the time of writing there are three atomics mappings (ABIs) `defined for RISC-V <https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/master/riscv-elf.adoc#tag_riscv_atomic_abi-14-uleb128version>`__.  As of LLVM 19, LLVM defaults to "A6S", which is compatible with both the original "A6" and the future "A7" ABI. See `the psABI atomics document <https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/master/riscv-atomic.adoc>`__ for more information on these mappings.

Note that although the "A6S" mapping is used, the ELF attribute recording the mapping isn't currently emitted by default due to a bug causing a crash in older versions of binutils when processing files containing this attribute.

Experimental Extensions
=======================

LLVM supports (to various degrees) a number of experimental extensions.  All experimental extensions have ``experimental-`` as a prefix.  There is explicitly no compatibility promised between versions of the toolchain, and regular users are strongly advised *not* to make use of experimental extensions before they reach ratification.

The primary goal of experimental support is to assist in the process of ratification by providing an existence proof of an implementation, and simplifying efforts to validate the value of a proposed extension against large code bases.  Experimental extensions are expected to either transition to ratified status, or be eventually removed.  The decision on whether to accept an experimental extension is currently done on an entirely case by case basis; if you want to propose one, attending the bi-weekly RISC-V sync-up call is strongly advised.

``experimental-zalasr``
  LLVM implements the `0.0.5 draft specification <https://github.com/mehnadnerd/riscv-zalasr>`__.

``experimental-zicfilp``, ``experimental-zicfiss``
  LLVM implements the `1.0 release specification <https://github.com/riscv/riscv-cfi/releases/tag/v1.0>`__.

``experimental-zvbc32e``, ``experimental-zvkgs``
  LLVM implements the `0.7 release specification <https://github.com/user-attachments/files/16450464/riscv-crypto-spec-vector-extra_v0.0.7.pdf>`__.

``experimental-smctr``, ``experimental-ssctr``
  LLVM implements the `1.0-rc3 specification <https://github.com/riscv/riscv-control-transfer-records/releases/tag/v1.0_rc3>`__.

``experimental-svukte``
  LLVM implements the `0.3 draft specification <https://github.com/riscv/riscv-isa-manual/pull/1564>`__.

To use an experimental extension from `clang`, you must add `-menable-experimental-extensions` to the command line, and specify the exact version of the experimental extension you are using.  To use an experimental extension with LLVM's internal developer tools (e.g. `llc`, `llvm-objdump`, `llvm-mc`), you must prefix the extension name with `experimental-`.  Note that you don't need to specify the version with internal tools, and shouldn't include the `experimental-` prefix with `clang`.

Vendor Extensions
=================

Vendor extensions are extensions which are not standardized by RISC-V International, and are instead defined by a hardware vendor.  The term vendor extension roughly parallels the definition of a `non-standard` extension from Section 1.3 of the Volume I: RISC-V Unprivileged ISA specification.  In particular, we expect to eventually accept both `custom` extensions and `non-conforming` extensions.

Inclusion of a vendor extension will be considered on a case by case basis.  All proposals should be brought to the bi-weekly RISCV sync calls for discussion.  For a general idea of the factors likely to be considered, please see the `Clang documentation <https://clang.llvm.org/get_involved.html>`__.

It is our intention to follow the naming conventions described in `riscv-non-isa/riscv-toolchain-conventions <https://github.com/riscv-non-isa/riscv-toolchain-conventions#conventions-for-vendor-extensions>`__.  Exceptions to this naming will need to be strongly motivated.

The current vendor extensions supported are:

``XTHeadBa``
  LLVM implements `the THeadBa (address-generation) vendor-defined instructions specified in <https://github.com/T-head-Semi/thead-extension-spec/releases/download/2.2.2/xthead-2023-01-30-2.2.2.pdf>`__ by T-HEAD of Alibaba.  Instructions are prefixed with `th.` as described in the specification.

``XTHeadBb``
  LLVM implements `the THeadBb (basic bit-manipulation) vendor-defined instructions specified in <https://github.com/T-head-Semi/thead-extension-spec/releases/download/2.2.2/xthead-2023-01-30-2.2.2.pdf>`__ by T-HEAD of Alibaba.  Instructions are prefixed with `th.` as described in the specification.

``XTHeadBs``
  LLVM implements `the THeadBs (single-bit operations) vendor-defined instructions specified in <https://github.com/T-head-Semi/thead-extension-spec/releases/download/2.2.2/xthead-2023-01-30-2.2.2.pdf>`__ by T-HEAD of Alibaba.  Instructions are prefixed with `th.` as described in the specification.

``XTHeadCondMov``
  LLVM implements `the THeadCondMov (conditional move) vendor-defined instructions specified in <https://github.com/T-head-Semi/thead-extension-spec/releases/download/2.2.2/xthead-2023-01-30-2.2.2.pdf>`__ by T-HEAD of Alibaba.  Instructions are prefixed with `th.` as described in the specification.

``XTHeadCmo``
  LLVM implements `the THeadCmo (cache management operations) vendor-defined instructions specified in <https://github.com/T-head-Semi/thead-extension-spec/releases/download/2.2.2/xthead-2023-01-30-2.2.2.pdf>`__  by T-HEAD of Alibaba.  Instructions are prefixed with `th.` as described in the specification.

``XTHeadFMemIdx``
  LLVM implements `the THeadFMemIdx (indexed memory operations for floating point) vendor-defined instructions specified in <https://github.com/T-head-Semi/thead-extension-spec/releases/download/2.2.2/xthead-2023-01-30-2.2.2.pdf>`__ by T-HEAD of Alibaba.  Instructions are prefixed with `th.` as described in the specification.

``XTheadMac``
  LLVM implements `the XTheadMac (multiply-accumulate instructions) vendor-defined instructions specified in <https://github.com/T-head-Semi/thead-extension-spec/releases/download/2.2.2/xthead-2023-01-30-2.2.2.pdf>`__ by T-HEAD of Alibaba.  Instructions are prefixed with `th.` as described in the specification.

``XTHeadMemIdx``
  LLVM implements `the THeadMemIdx (indexed memory operations) vendor-defined instructions specified in <https://github.com/T-head-Semi/thead-extension-spec/releases/download/2.2.2/xthead-2023-01-30-2.2.2.pdf>`__ by T-HEAD of Alibaba.  Instructions are prefixed with `th.` as described in the specification.

``XTHeadMemPair``
  LLVM implements `the THeadMemPair (two-GPR memory operations) vendor-defined instructions specified in <https://github.com/T-head-Semi/thead-extension-spec/releases/download/2.2.2/xthead-2023-01-30-2.2.2.pdf>`__ by T-HEAD of Alibaba.  Instructions are prefixed with `th.` as described in the specification.

``XTHeadSync``
  LLVM implements `the THeadSync (multi-core synchronization instructions) vendor-defined instructions specified in <https://github.com/T-head-Semi/thead-extension-spec/releases/download/2.2.2/xthead-2023-01-30-2.2.2.pdf>`__ by T-HEAD of Alibaba.  Instructions are prefixed with `th.` as described in the specification.

``XTHeadVdot``
  LLVM implements `version 1.0.0 of the THeadV-family custom instructions specification <https://github.com/T-head-Semi/thead-extension-spec/releases/download/2.2.0/xthead-2022-12-04-2.2.0.pdf>`__ by T-HEAD of Alibaba.  All instructions are prefixed with `th.` as described in the specification, and the riscv-toolchain-convention document linked above.

``XVentanaCondOps``
  LLVM implements `version 1.0.0 of the VTx-family custom instructions specification <https://github.com/ventanamicro/ventana-custom-extensions/releases/download/v1.0.0/ventana-custom-extensions-v1.0.0.pdf>`__ by Ventana Micro Systems.  All instructions are prefixed with `vt.` as described in the specification, and the riscv-toolchain-convention document linked above.  These instructions are only available for riscv64 at this time.

``XSfvcp``
  LLVM implements `version 1.1.0 of the SiFive Vector Coprocessor Interface (VCIX) Software Specification <https://sifive.cdn.prismic.io/sifive/Zn3m1R5LeNNTwnLS_vcix-spec-software-v1p1.pdf>`__ by SiFive.  All instructions are prefixed with `sf.vc.` as described in the specification, and the riscv-toolchain-convention document linked above.

``XSfvqmaccdod``, ``XSfvqmaccqoq``
  LLVM implements `version 1.1.0 of the SiFive Int8 Matrix Multiplication Extensions Specification <https://sifive.cdn.prismic.io/sifive/1a2ad85b-d818-49f7-ba83-f51f1731edbe_int8-matmul-spec.pdf>`__ by SiFive.  All instructions are prefixed with `sf.` as described in the specification linked above.

``Xsfvfnrclipxfqf``
  LLVM implements `version 1.0.0 of the FP32-to-int8 Ranged Clip Instructions Extension Specification <https://sifive.cdn.prismic.io/sifive/0aacff47-f530-43dc-8446-5caa2260ece0_xsfvfnrclipxfqf-spec.pdf>`__ by SiFive.  All instructions are prefixed with `sf.` as described in the specification linked above.

``Xsfvfwmaccqqq``
  LLVM implements `version 1.0.0 of the Matrix Multiply Accumulate Instruction Extension Specification <https://sifive.cdn.prismic.io/sifive/c391d53e-ffcf-4091-82f6-c37bf3e883ed_xsfvfwmaccqqq-spec.pdf>`__ by SiFive.  All instructions are prefixed with `sf.` as described in the specification linked above.

``XCVbitmanip``
  LLVM implements `version 1.0.0 of the CORE-V Bit Manipulation custom instructions specification <https://github.com/openhwgroup/cv32e40p/blob/62bec66b36182215e18c9cf10f723567e23878e9/docs/source/instruction_set_extensions.rst>`__ by OpenHW Group.  All instructions are prefixed with `cv.` as described in the specification.

``XCVelw``
  LLVM implements `version 1.0.0 of the CORE-V Event load custom instructions specification <https://github.com/openhwgroup/cv32e40p/blob/master/docs/source/instruction_set_extensions.rst>`__ by OpenHW Group.  All instructions are prefixed with `cv.` as described in the specification. These instructions are only available for riscv32 at this time.

``XCVmac``
  LLVM implements `version 1.0.0 of the CORE-V Multiply-Accumulate (MAC) custom instructions specification <https://github.com/openhwgroup/cv32e40p/blob/4f024fe4b15a68b76615b0630c07a6745c620da7/docs/source/instruction_set_extensions.rst>`__ by OpenHW Group.  All instructions are prefixed with `cv.mac` as described in the specification. These instructions are only available for riscv32 at this time.

``XCVmem``
  LLVM implements `version 1.0.0 of the CORE-V Post-Increment load and stores custom instructions specification <https://github.com/openhwgroup/cv32e40p/blob/master/docs/source/instruction_set_extensions.rst>`__ by OpenHW Group.  All instructions are prefixed with `cv.` as described in the specification. These instructions are only available for riscv32 at this time.

``XCValu``
  LLVM implements `version 1.0.0 of the Core-V ALU custom instructions specification <https://github.com/openhwgroup/cv32e40p/blob/4f024fe4b15a68b76615b0630c07a6745c620da7/docs/source/instruction_set_extensions.rst>`__ by Core-V.  All instructions are prefixed with `cv.` as described in the specification. These instructions are only available for riscv32 at this time.

``XCVsimd``
  LLVM implements `version 1.0.0 of the CORE-V SIMD custom instructions specification <https://github.com/openhwgroup/cv32e40p/blob/cv32e40p_v1.3.2/docs/source/instruction_set_extensions.rst>`__ by OpenHW Group.  All instructions are prefixed with `cv.` as described in the specification.

``XCVbi``
  LLVM implements `version 1.0.0 of the CORE-V immediate branching custom instructions specification <https://github.com/openhwgroup/cv32e40p/blob/cv32e40p_v1.3.2/docs/source/instruction_set_extensions.rst>`__ by OpenHW Group.  All instructions are prefixed with `cv.` as described in the specification. These instructions are only available for riscv32 at this time.

``XSiFivecdiscarddlone``
  LLVM implements `the SiFive sf.cdiscard.d.l1 instruction specified in <https://sifive.cdn.prismic.io/sifive/767804da-53b2-4893-97d5-b7c030ae0a94_s76mc_core_complex_manual_21G3.pdf>`_ by SiFive.

``XSiFivecflushdlone``
  LLVM implements `the SiFive sf.cflush.d.l1 instruction specified in <https://sifive.cdn.prismic.io/sifive/767804da-53b2-4893-97d5-b7c030ae0a94_s76mc_core_complex_manual_21G3.pdf>`_ by SiFive.

``XSfcease``
  LLVM implements `the SiFive sf.cease instruction specified in <https://sifive.cdn.prismic.io/sifive/767804da-53b2-4893-97d5-b7c030ae0a94_s76mc_core_complex_manual_21G3.pdf>`_ by SiFive.

``Xwchc``
  LLVM implements `the custom compressed opcodes present in some QingKe cores` by WCH / Nanjing Qinheng Microelectronics. The vendor refers to these opcodes by the name "XW".

``experimental-Xqcia``
  LLVM implements `version 0.2 of the Qualcomm uC Arithmetic extension specification <https://github.com/quic/riscv-unified-db/releases/latest>`__ by Qualcomm.  All instructions are prefixed with `qc.` as described in the specification. These instructions are only available for riscv32.

``experimental-Xqcics``
  LLVM implements `version 0.2 of the Qualcomm uC Conditional Select extension specification <https://github.com/quic/riscv-unified-db/releases/latest>`__ by Qualcomm.  All instructions are prefixed with `qc.` as described in the specification. These instructions are only available for riscv32.

``experimental-Xqcicsr``
  LLVM implements `version 0.2 of the Qualcomm uC CSR extension specification <https://github.com/quic/riscv-unified-db/releases/latest>`__ by Qualcomm.  All instructions are prefixed with `qc.` as described in the specification. These instructions are only available for riscv32.

``experimental-Xqcilsm``
  LLVM implements `version 0.2 of the Qualcomm uC Load Store Multiple extension specification <https://github.com/quic/riscv-unified-db/releases/latest>`__ by Qualcomm.  All instructions are prefixed with `qc.` as described in the specification. These instructions are only available for riscv32.

``experimental-Xqcisls``
  LLVM implements `version 0.2 of the Qualcomm uC Scaled Load Store extension specification <https://github.com/quic/riscv-unified-db/releases/latest>`__ by Qualcomm.  All instructions are prefixed with `qc.` as described in the specification. These instructions are only available for riscv32.

Experimental C Intrinsics
=========================

In some cases an extension is non-experimental but the C intrinsics for that
extension are still experimental.  To use C intrinsics for such an extension
from `clang`, you must add `-menable-experimental-extensions` to the command
line.  This currently applies to the following extensions:

No extensions have experimental intrinsics.

Long (>32-bit) Instruction Support
==================================

RISC-V is a variable-length ISA, but the standard currently only defines 16- and 32-bit instructions. The specification describes longer instruction encodings, but these are not ratified.

The LLVM disassembler, `llvm-objdump`, does use the longer instruction encodings described in the specification to guess the instruction length (up to 176 bits) and will group the disassembly view of encoding bytes correspondingly.

The LLVM integrated assembler for RISC-V supports two different kinds of ``.insn`` directive, for assembling instructions that LLVM does not yet support:

* ``.insn type, args*`` which takes a known instruction type, and a list of fields. You are strongly recommended to use this variant of the directive if your instruction fits an existing instruction type.
* ``.insn [ length , ] encoding`` which takes an (optional) explicit length (in bytes) and a raw encoding for the instruction. When given an explicit length, this variant can encode instructions up to 64 bits long. The encoding part of the directive must be given all bits for the instruction, none are filled in for the user. When used without the optional length, this variant of the directive will use the LSBs of the raw encoding to work out if an instruction is 16 or 32 bits long. LLVM does not infer that an instruction might be longer than 32 bits - in this case, the user must give the length explicitly.

It is strongly recommended to use the ``.insn`` directive for assembling unsupported instructions instead of ``.word`` or ``.hword``, because it will produce the correct mapping symbols to mark the word as an instruction, not data.

Global Pointer (GP) Relaxation and the Small Data Limit
=======================================================

Some of the RISC-V psABI variants reserve ``gp`` (``x3``) for use as a "Global Pointer", to make generating data addresses more efficient.

To use this functionality, you need to be doing all of the following:

* Use the ``medlow`` (aka ``small``) code model;
* Not use the ``gp`` register for any other uses (some platforms use it for the shadow stack and others as a temporary -- as denoted by the ``Tag_RISCV_x3_reg_usage`` build attribute);
* Compile your objects with Clang's ``-mrelax`` option, to enable relaxation annotations on relocatable objects (this is the default, but ``-mno-relax`` disables these relaxation annotations);
* Compile for a position-dependent static executable (not a shared library, and ``-fno-PIC`` / ``-fno-pic`` / ``-fno-pie``); and
* Use LLD's ``--relax-gp`` option.

LLD will relax (rewrite) any code sequences that materialize an address within 2048 bytes of ``__global_pointer$`` (which will be defined if it is used and does not already exist) to instead generate the address using ``gp`` and the correct (signed) 12-bit immediate. This usually saves at least one instruction compared to materialising a full 32-bit address value.

There can only be one ``gp`` value in a process (as ``gp`` is not changed when calling into a function in a shared library), so the symbol is is only defined and this relaxation is only done for executables, and not for shared libraries. The linker expects executable startup code to put the value of ``__global_pointer$`` (from the executable) into ``gp`` before any user code is run.

Arguably, the most efficient use for this addressing mode is for smaller global variables, as larger global variables likely need many more loads or stores when they are being accessed anyway, so the cost of materializing the upper bits can be shared.

Therefore the compiler can place smaller global variables into sections with names starting with ``.sdata`` or ``.sbss`` (matching sections with names starting with ``.data`` and ``.bss`` respectively). LLD knows to define the ``global_pointer$`` symbol close to these sections, and to lay these sections out adjacent to the ``.data`` section.

Clang's ``-msmall-data-limit=`` option controls what the threshold size is (in bytes) for a global variable to be considered small. ``-msmall-data-limit=0`` disables the use of sections starting ``.sdata`` and ``.sbss``. The ``-msmall-data-limit=`` option will not move global variables that have an explicit data section, and will keep globals in separate sections if you are using ``-fdata-sections``.

The small data limit threshold is also used to separate small constants into sections with names starting with ``.srodata``. LLD does not place these with the ``.sdata`` and ``.sbss`` sections as ``.srodata`` sections are read only and the other two are writable. Instead the ``.srodata`` sections are placed adjacent to ``.rodata``.

Data suggests that these options can produce significant improvements across a range of benchmarks.
