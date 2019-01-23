========================
Scudo Hardened Allocator
========================

.. contents::
   :local:
   :depth: 1

Introduction
============

The Scudo Hardened Allocator is a user-mode allocator based on LLVM Sanitizer's
CombinedAllocator, which aims at providing additional mitigations against heap
based vulnerabilities, while maintaining good performance.

Currently, the allocator supports (was tested on) the following architectures:

- i386 (& i686) (32-bit);
- x86_64 (64-bit);
- armhf (32-bit);
- AArch64 (64-bit);
- MIPS (32-bit & 64-bit).

The name "Scudo" has been retained from the initial implementation (Escudo
meaning Shield in Spanish and Portuguese).

Design
======

Allocator
---------
Scudo can be considered a Frontend to the Sanitizers' common allocator (later
referenced as the Backend). It is split between a Primary allocator, fast and
efficient, that services smaller allocation sizes, and a Secondary allocator
that services larger allocation sizes and is backed by the operating system
memory mapping primitives.

Scudo was designed with security in mind, but aims at striking a good balance
between security and performance. It is highly tunable and configurable.

Chunk Header
------------
Every chunk of heap memory will be preceded by a chunk header. This has two
purposes, the first one being to store various information about the chunk,
the second one being to detect potential heap overflows. In order to achieve
this, the header will be checksummed, involving the pointer to the chunk itself
and a global secret. Any corruption of the header will be detected when said
header is accessed, and the process terminated.

The following information is stored in the header:

- the 16-bit checksum;
- the class ID for that chunk, which is the "bucket" where the chunk resides
  for Primary backed allocations, or 0 for Secondary backed allocations;
- the size (Primary) or unused bytes amount (Secondary) for that chunk, which is
  necessary for computing the size of the chunk;
- the state of the chunk (available, allocated or quarantined);
- the allocation type (malloc, new, new[] or memalign), to detect potential
  mismatches in the allocation APIs used;
- the offset of the chunk, which is the distance in bytes from the beginning of
  the returned chunk to the beginning of the Backend allocation;

This header fits within 8 bytes, on all platforms supported.

The checksum is computed as a CRC32 (made faster with hardware support)
of the global secret, the chunk pointer itself, and the 8 bytes of header with
the checksum field zeroed out. It is not intended to be cryptographically
strong. 

The header is atomically loaded and stored to prevent races. This is important
as two consecutive chunks could belong to different threads. We also want to
avoid any type of double fetches of information located in the header, and use
local copies of the header for this purpose.

Delayed Freelist
-----------------
A delayed freelist allows us to not return a chunk directly to the Backend, but
to keep it aside for a while. Once a criterion is met, the delayed freelist is
emptied, and the quarantined chunks are returned to the Backend. This helps
mitigate use-after-free vulnerabilities by reducing the determinism of the
allocation and deallocation patterns.

This feature is using the Sanitizer's Quarantine as its base, and the amount of
memory that it can hold is configurable by the user (see the Options section
below).

Randomness
----------
It is important for the allocator to not make use of fixed addresses. We use
the dynamic base option for the SizeClassAllocator, allowing us to benefit
from the randomness of the system memory mapping functions.

Usage
=====

Library
-------
The allocator static library can be built from the LLVM build tree thanks to
the ``scudo`` CMake rule. The associated tests can be exercised thanks to the
``check-scudo`` CMake rule.

Linking the static library to your project can require the use of the
``whole-archive`` linker flag (or equivalent), depending on your linker.
Additional flags might also be necessary.

Your linked binary should now make use of the Scudo allocation and deallocation
functions.

You may also build Scudo like this: 

.. code:: console

  cd $LLVM/projects/compiler-rt/lib
  clang++ -fPIC -std=c++11 -msse4.2 -O2 -I. scudo/*.cpp \
    $(\ls sanitizer_common/*.{cc,S} | grep -v "sanitizer_termination\|sanitizer_common_nolibc\|sancov_\|sanitizer_unwind\|sanitizer_symbol") \
    -shared -o libscudo.so -pthread

and then use it with existing binaries as follows:

.. code:: console

  LD_PRELOAD=`pwd`/libscudo.so ./a.out

Clang
-----
With a recent version of Clang (post rL317337), the allocator can be linked with
a binary at compilation using the ``-fsanitize=scudo`` command-line argument, if
the target platform is supported. Currently, the only other Sanitizer Scudo is
compatible with is UBSan (eg: ``-fsanitize=scudo,undefined``). Compiling with
Scudo will also enforce PIE for the output binary.

Options
-------
Several aspects of the allocator can be configured on a per process basis
through the following ways:

- at compile time, by defining ``SCUDO_DEFAULT_OPTIONS`` to the options string
  you want set by default;

- by defining a ``__scudo_default_options`` function in one's program that
  returns the options string to be parsed. Said function must have the following
  prototype: ``extern "C" const char* __scudo_default_options(void)``, with a
  default visibility. This will override the compile time define;

- through the environment variable SCUDO_OPTIONS, containing the options string
  to be parsed. Options defined this way will override any definition made
  through ``__scudo_default_options``.

The options string follows a syntax similar to ASan, where distinct options
can be assigned in the same string, separated by colons.

For example, using the environment variable:

.. code:: console

  SCUDO_OPTIONS="DeleteSizeMismatch=1:QuarantineSizeKb=64" ./a.out

Or using the function:

.. code:: cpp

  extern "C" const char *__scudo_default_options() {
    return "DeleteSizeMismatch=1:QuarantineSizeKb=64";
  }


The following options are available:

+-----------------------------+----------------+----------------+------------------------------------------------+
| Option                      | 64-bit default | 32-bit default | Description                                    |
+-----------------------------+----------------+----------------+------------------------------------------------+
| QuarantineSizeKb            | 256            | 64             | The size (in Kb) of quarantine used to delay   |
|                             |                |                | the actual deallocation of chunks. Lower value |
|                             |                |                | may reduce memory usage but decrease the       |
|                             |                |                | effectiveness of the mitigation; a negative    |
|                             |                |                | value will fallback to the defaults. Setting   |
|                             |                |                | *both* this and ThreadLocalQuarantineSizeKb to |
|                             |                |                | zero will disable the quarantine entirely.     |
+-----------------------------+----------------+----------------+------------------------------------------------+
| QuarantineChunksUpToSize    | 2048           | 512            | Size (in bytes) up to which chunks can be      |
|                             |                |                | quarantined.                                   |
+-----------------------------+----------------+----------------+------------------------------------------------+
| ThreadLocalQuarantineSizeKb | 1024           | 256            | The size (in Kb) of per-thread cache use to    |
|                             |                |                | offload the global quarantine. Lower value may |
|                             |                |                | reduce memory usage but might increase         |
|                             |                |                | contention on the global quarantine. Setting   |
|                             |                |                | *both* this and QuarantineSizeKb to zero will  |
|                             |                |                | disable the quarantine entirely.               |
+-----------------------------+----------------+----------------+------------------------------------------------+
| DeallocationTypeMismatch    | true           | true           | Whether or not we report errors on             |
|                             |                |                | malloc/delete, new/free, new/delete[], etc.    |
+-----------------------------+----------------+----------------+------------------------------------------------+
| DeleteSizeMismatch          | true           | true           | Whether or not we report errors on mismatch    |
|                             |                |                | between sizes of new and delete.               |
+-----------------------------+----------------+----------------+------------------------------------------------+
| ZeroContents                | false          | false          | Whether or not we zero chunk contents on       |
|                             |                |                | allocation and deallocation.                   |
+-----------------------------+----------------+----------------+------------------------------------------------+

Allocator related common Sanitizer options can also be passed through Scudo
options, such as ``allocator_may_return_null`` or ``abort_on_error``. A detailed
list including those can be found here:
https://github.com/google/sanitizers/wiki/SanitizerCommonFlags.

Error Types
===========

The allocator will output an error message, and potentially terminate the
process, when an unexpected behavior is detected. The output usually starts with
``"Scudo ERROR:"`` followed by a short summary of the problem that occurred as
well as the pointer(s) involved. Once again, Scudo is meant to be a mitigation,
and might not be the most useful of tools to help you root-cause the issue,
please consider `ASan <https://github.com/google/sanitizers/wiki/AddressSanitizer>`_
for this purpose.

Here is a list of the current error messages and their potential cause:

- ``"corrupted chunk header"``: the checksum verification of the chunk header
  has failed. This is likely due to one of two things: the header was
  overwritten (partially or totally), or the pointer passed to the function is
  not a chunk at all;

- ``"race on chunk header"``: two different threads are attempting to manipulate
  the same header at the same time. This is usually symptomatic of a
  race-condition or general lack of locking when performing operations on that
  chunk;

- ``"invalid chunk state"``: the chunk is not in the expected state for a given
  operation, eg: it is not allocated when trying to free it, or it's not
  quarantined when trying to recycle it, etc. A double-free is the typical
  reason this error would occur;

- ``"misaligned pointer"``: we strongly enforce basic alignment requirements, 8
  bytes on 32-bit platforms, 16 bytes on 64-bit platforms. If a pointer passed
  to our functions does not fit those, something is definitely wrong.

- ``"allocation type mismatch"``: when the optional deallocation type mismatch
  check is enabled, a deallocation function called on a chunk has to match the
  type of function that was called to allocate it. Security implications of such
  a mismatch are not necessarily obvious but situational at best;

- ``"invalid sized delete"``: when the C++14 sized delete operator is used, and
  the optional check enabled, this indicates that the size passed when
  deallocating a chunk is not congruent with the one requested when allocating
  it. This is likely to be a `compiler issue <https://software.intel.com/en-us/forums/intel-c-compiler/topic/783942>`_,
  as was the case with Intel C++ Compiler, or some type confusion on the object
  being deallocated;

- ``"RSS limit exhausted"``: the maximum RSS optionally specified has been
  exceeded;

Several other error messages relate to parameter checking on the libc allocation
APIs and are fairly straightforward to understand.
