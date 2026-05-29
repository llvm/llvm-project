=========================
The ``.metallib`` Format
=========================

.. contents::
   :local:

Status
======

Apple does not publish a specification for the ``.metallib`` container or
for the AIR bitcode dialect it carries. The layout described here was
reconstructed empirically from the output of Apple's Metal toolchain and is
byte-identical to it for the kernels in the LLVM Metal test suite. Field
names mirror the implementation in
``llvm/lib/Target/Metal/AIRWriter/MetallibWriter.{h,cpp}``.

All integers are little-endian.

Top-Level Layout
================

A ``.metallib`` file is laid out as::

    [   0 ..  88) MTLB header
    [  88 ..  ? ) Section 0 — entry table
    [   ? ..  ? ) ENDT gap (2 × 4 bytes per entry; not counted in section 0)
    [   ? ..  ? ) Section 1 — function list  (u32(4) + "ENDT")
    [   ? ..  ? ) Section 2 — public metadata (u32(4) + "ENDT")
    [   ? .. EOF) Section 3 — wrapped AIR bitcode

The header carries four ``(offset, size)`` pairs locating sections 0–3.

MTLB Header
===========

The 88-byte header is::

    offset   size   field            value / meaning
    ------   ----   --------------   --------------------------------
       0       4   Magic            ``"MTLB"``
       4       2   FormatMinor      ``0x0180`` (little-endian: 0x80 0x01)
       6       2   FormatMajor      ``0x0002``
       8       1   Reserved         ``0x09``
       9       3   Padding          zero
      12       1   Reserved         zero
      13       1   Platform         ``0x81`` (macOS) or ``0x82`` (iOS)
      14       1   OSMajor          (e.g. ``26`` — see DefaultMetalLibOSMajor)
      15       3   OSPad            zero
      20       8   FileSize         total size of the file in bytes
      28       8   Sec0Offset
      36       8   Sec0Size
      44       8   Sec1Offset
      52       8   Sec1Size
      60       8   Sec2Offset
      68       8   Sec2Size
      76       8   Sec3Offset
      84       8   Sec3Size         (== wrapped bitcode size)

Section 0 — Entry Table
=======================

Section 0 begins with::

    u32 EntryCount;

followed by ``EntryCount`` entry blocks. Each entry block is::

    u32 EntrySize;       // size of the tags that follow (excludes ENDT pair)
    Tag tags[];          // see "Tag format" below

After section 0, a gap of ``EntryCount × 8`` bytes contains two ``"ENDT"``
literals per entry. These ENDT terminators sit *outside* the recorded
section-0 size — readers must therefore advance ``EntrySize`` bytes past
``EntrySize`` itself when walking entries, and skip ``EntryCount × 8``
extra bytes to reach section 1.

Tag Format
----------

Each tag inside an entry block is::

    char Name[4];   // four ASCII letters
    u16  Length;
    u8   Payload[Length];

The ``ENDT`` terminator is the sole exception: it is four bytes (``"ENDT"``)
with no length field.

Tags emitted per entry, in order:

============ ====================== =======================================
Tag          Payload size           Contents
============ ====================== =======================================
``NAME``     ``len(name) + 1``      NUL-terminated kernel name
``TYPE``     1                      ``2`` (kernel)
``HASH``     32                     SHA-256 of the wrapped bitcode
``MDSZ``     8                      Bitcode section size, u64
``OFFT``     24                     Three u64 offsets, zero for single entry
``VERS``     8                      ``u16{air_major=2, air_minor=8,``
                                    ``metal_major, metal_minor}``
============ ====================== =======================================

Sections 1 and 2
================

Sections 1 (function list) and 2 (public metadata) are each four bytes
followed by the ``ENDT`` literal (8 bytes total per section)::

    u32 PayloadSize;   // 4
    char Terminator[4]; // "ENDT"

For the kernels in the test suite, both sections are placeholders.

Section 3 — Wrapped Bitcode
===========================

Section 3 holds the AIR bitcode behind LLVM's standard 20-byte bitcode
wrapper::

    u32 WrapperMagic;   // 0x0B17C0DE
    u32 Version;        // 0
    u32 BitcodeOffset;  // 20
    u32 BitcodeSize;
    u32 CpuType;        // 0xFFFFFFFF
    u8  Bitcode[BitcodeSize];

The bitcode itself is AIR bitcode emitted by ``BitcodeEmitter`` — a subset
of LLVM bitcode that uses typed pointer records (``POINTER`` code 8)
instead of the opaque-pointer records (code 25) that LLVM 17+ writes by
default. The AIR module datalayout, triple, and intrinsic names are
documented in :doc:`MetalTarget`.

Reading a ``.metallib``
=======================

Today, the in-tree pieces are write-only:

* ``MetalEmbedderPass`` builds the metallib bytes from a Module.
* ``MetalLibObjectWriter`` (``llvm/lib/MC/MetalLibObjectWriter.cpp``) wraps
  those bytes into ``-filetype=obj`` output.
* ``MCMetalLibStreamer`` (``llvm/lib/MC/MCMetalLibStreamer.cpp``) plumbs
  the embedder output through the MC streamer interface.

There is no corresponding parser; introspecting a ``.metallib`` is left to
``llvm-bcanalyzer`` (on the unwrapped bitcode payload) and to external
tooling.
