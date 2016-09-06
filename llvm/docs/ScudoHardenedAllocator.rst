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

The name "Scudo" has been retained from the initial implementation (Escudo
meaning Shield in Spanish and Portuguese).

Design
======

Chunk Header
------------
Every chunk of heap memory will be preceded by a chunk header. This has two
purposes, the first one being to store various information about the chunk,
the second one being to detect potential heap overflows. In order to achieve
this, the header will be checksumed, involving the pointer to the chunk itself
and a global secret. Any corruption of the header will be detected when said
header is accessed, and the process terminated.

The following information is stored in the header:

- the 16-bit checksum;
- the user requested size for that chunk, which is necessary for reallocation
  purposes;
- the state of the chunk (available, allocated or quarantined);
- the allocation type (malloc, new, new[] or memalign), to detect potential
  mismatches in the allocation APIs used;
- whether or not the chunk is offseted (ie: if the chunk beginning is different
  than the backend allocation beginning, which is most often the case with some
  aligned allocations);
- the associated offset;
- a 16-bit salt.

On x64, which is currently the only architecture supported, the header fits
within 16-bytes, which works nicely with the minimum alignment requirements.

The checksum is computed as a CRC32 (requiring the SSE 4.2 instruction set)
of the global secret, the chunk pointer itself, and the 16 bytes of header with
the checksum field zeroed out.

The header is atomically loaded and stored to prevent races (this requires
platform support such as the cmpxchg16b instruction). This is important as two
consecutive chunks could belong to different threads. We also want to avoid
any type of double fetches of information located in the header, and use local
copies of the header for this purpose.

Delayed Freelist
-----------------
A delayed freelist allows us to not return a chunk directly to the backend, but
to keep it aside for a while. Once a criterion is met, the delayed freelist is
emptied, and the quarantined chunks are returned to the backend. This helps
mitigate use-after-free vulnerabilities by reducing the determinism of the
allocation and deallocation patterns.

This feature is using the Sanitizer's Quarantine as its base, and the amount of
memory that it can hold is configurable by the user (see the Options section
below).

Randomness
----------
It is important for the allocator to not make use of fixed addresses. We use
the dynamic base option for the SizeClassAllocator, allowing us to benefit
from the randomness of mmap.

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

.. code::

  cd $LLVM/projects/compiler-rt/lib
  clang++ -fPIC -std=c++11 -msse4.2 -mcx16 -O2 -I. scudo/*.cpp \
    $(\ls sanitizer_common/*.{cc,S} | grep -v "sanitizer_termination\|sanitizer_common_nolibc") \
    -shared -o scudo-allocator.so -lpthread

and then use it with existing binaries as follows:

.. code::

  LD_PRELOAD=`pwd`/scudo-allocator.so ./a.out

Options
-------
Several aspects of the allocator can be configured through the following ways:

- by defining a ``__scudo_default_options`` function in one's program that
  returns the options string to be parsed. Said function must have the following
  prototype: ``extern "C" const char* __scudo_default_options()``.

- through the environment variable SCUDO_OPTIONS, containing the options string
  to be parsed. Options defined this way will override any definition made
  through ``__scudo_default_options``;

The options string follows a syntax similar to ASan, where distinct options
can be assigned in the same string, separated by colons.

For example, using the environment variable:

.. code::

  SCUDO_OPTIONS="DeleteSizeMismatch=1:QuarantineSizeMb=16" ./a.out

Or using the function:

.. code::

  extern "C" const char *__scudo_default_options() {
    return "DeleteSizeMismatch=1:QuarantineSizeMb=16";
  }


The following options are available:

+-----------------------------+---------+------------------------------------------------+
| Option                      | Default | Description                                    |
+-----------------------------+---------+------------------------------------------------+
| QuarantineSizeMb            | 64      | The size (in Mb) of quarantine used to delay   |
|                             |         | the actual deallocation of chunks. Lower value |
|                             |         | may reduce memory usage but decrease the       |
|                             |         | effectiveness of the mitigation; a negative    |
|                             |         | value will fallback to a default of 64Mb.      |
+-----------------------------+---------+------------------------------------------------+
| ThreadLocalQuarantineSizeKb | 1024    | The size (in Kb) of per-thread cache use to    |
|                             |         | offload the global quarantine. Lower value may |
|                             |         | reduce memory usage but might increase         |
|                             |         | contention on the global quarantine.           |
+-----------------------------+---------+------------------------------------------------+
| DeallocationTypeMismatch    | true    | Whether or not we report errors on             |
|                             |         | malloc/delete, new/free, new/delete[], etc.    |
+-----------------------------+---------+------------------------------------------------+
| DeleteSizeMismatch          | true    | Whether or not we report errors on mismatch    |
|                             |         | between sizes of new and delete.               |
+-----------------------------+---------+------------------------------------------------+
| ZeroContents                | false   | Whether or not we zero chunk contents on       |
|                             |         | allocation and deallocation.                   |
+-----------------------------+---------+------------------------------------------------+

Allocator related common Sanitizer options can also be passed through Scudo
options, such as ``allocator_may_return_null``. A detailed list including those
can be found here:
https://github.com/google/sanitizers/wiki/SanitizerCommonFlags.

