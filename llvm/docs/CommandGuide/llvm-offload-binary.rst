llvm-offload-binary - LLVM Offload Binary Packager
==================================================

.. program:: llvm-offload-binary

SYNOPSIS
--------

:program:`llvm-offload-binary` [*options*] [*input files...*]

DESCRIPTION
-----------

:program:`llvm-offload-binary` is a utility for bundling multiple device object
files into a single binary container. The resulting binary can then be embedded
into the host section table to form a fat binary containing offloading code for
different targets. Conversely, it can also extract previously bundled device
images.

The binary format begins with the magic bytes ``0x10FF10AD``, followed by a
version and size. Each binary contains its own header, allowing tools to locate
offloading sections even when merged by a linker. Each offload entry includes
metadata such as the device image kind, producer kind, and key-value string
metadata. Multiple offloading images are concatenated to form a fat binary.

EXAMPLE
-------

.. code-block:: console

  # Package multiple device images into a fat binary:
  $ llvm-offload-binary -o out.bin \
        --image=file=input.o,triple=nvptx64,arch=sm_70

  # Extract a matching image from a fat binary:
  $ llvm-offload-binary in.bin \
        --image=file=output.o,triple=nvptx64,arch=sm_70

  # Extract and archive images into a static library:
  $ llvm-offload-binary in.bin --archive -o libdevice.a

OPTIONS
-------

.. option:: --archive

  When extracting from an input binary, write all extracted images into a static
  archive instead of separate files.

.. option:: --image=<<key>=<value>,...>

  Specify a set of arbitrary key-value arguments describing an image.
  Commonly used optional keys include ``arch`` (e.g. ``sm_70`` for CUDA) and
  ``triple`` (e.g. nvptx64-nvidia-cuda).

.. option:: -o <file>

  Write output to <file>. When bundling, this specifies the fat binary filename.
  When extracting, this specifies the archive or output file destination.

.. option:: --help, -h

  Display available options. Use ``--help-hidden`` to show hidden options.

.. option:: --help-list

  Display a list of all options. Use ``--help-list-hidden`` to show hidden ones.

.. option:: --version

  Display the version of the :program:`llvm-offload-binary` executable.

.. option:: @<FILE>

  Read command-line options from response file `<FILE>`.

BINARY FORMAT
-------------

The binary format is marked by the magic bytes ``0x10FF10AD``, followed by a
version number. Each created binary contains its own header. This allows tools
to locate offloading sections even after linker operations such as relocatable
linking. Conceptually, this binary format is a serialization of a string map and
an image buffer.

.. table:: Offloading Binary Header
   :name: table-binary_header

   +----------+--------------+----------------------------------------------------+
   |   Type   |  Identifier  | Description                                        |
   +==========+==============+====================================================+
   | uint8_t  |    magic     | The magic bytes for the binary format (0x10FF10AD) |
   +----------+--------------+----------------------------------------------------+
   | uint32_t |   version    | Version of this format (currently version 1)       |
   +----------+--------------+----------------------------------------------------+
   | uint64_t |    size      | Size of this binary in bytes                       |
   +----------+--------------+----------------------------------------------------+
   | uint64_t | entry offset | Absolute offset of the offload entries in bytes    |
   +----------+--------------+----------------------------------------------------+
   | uint64_t |  entry size  | Size of the offload entries in bytes               |
   +----------+--------------+----------------------------------------------------+

Each offload entry describes a bundled image along with its associated metadata.

.. table:: Offloading Entry Table
   :name: table-binary_entry

   +----------+---------------+----------------------------------------------------+
   |   Type   |   Identifier  | Description                                        |
   +==========+===============+====================================================+
   | uint16_t |  image kind   | The kind of the device image (e.g. bc, cubin)      |
   +----------+---------------+----------------------------------------------------+
   | uint16_t | offload kind  | The producer of the image (e.g. openmp, cuda)      |
   +----------+---------------+----------------------------------------------------+
   | uint32_t |     flags     | Generic flags for the image                        |
   +----------+---------------+----------------------------------------------------+
   | uint64_t | string offset | Absolute offset of the string metadata table       |
   +----------+---------------+----------------------------------------------------+
   | uint64_t |  num strings  | Number of string entries in the table              |
   +----------+---------------+----------------------------------------------------+
   | uint64_t |  image offset | Absolute offset of the device image in bytes       |
   +----------+---------------+----------------------------------------------------+
   | uint64_t |   image size  | Size of the device image in bytes                  |
   +----------+---------------+----------------------------------------------------+

The entry table refers to both a string table and the raw device image itself.
The string table provides arbitrary key-value metadata.

.. table:: Offloading String Entry
   :name: table-binary_string

   +----------+--------------+-------------------------------------------------------+
   |   Type   |   Identifier | Description                                           |
   +==========+==============+=======================================================+
   | uint64_t |  key offset  | Absolute byte offset of the key in the string table   |
   +----------+--------------+-------------------------------------------------------+
   | uint64_t | value offset | Absolute byte offset of the value in the string table |
   +----------+--------------+-------------------------------------------------------+

The string table is a collection of null-terminated strings stored in the image.
Offsets allow string entries to be interpreted as key-value pairs, enabling
flexible metadata such as architecture or target triple.

The enumerated values for ``image kind`` and ``offload kind`` are:

.. table:: Image Kind
   :name: table-image_kind

   +---------------+-------+---------------------------------------+
   |      Name     | Value | Description                           |
   +===============+=======+=======================================+
   | IMG_None      | 0x00  | No image information provided         |
   +---------------+-------+---------------------------------------+
   | IMG_Object    | 0x01  | The image is a generic object file    |
   +---------------+-------+---------------------------------------+
   | IMG_Bitcode   | 0x02  | The image is an LLVM-IR bitcode file  |
   +---------------+-------+---------------------------------------+
   | IMG_Cubin     | 0x03  | The image is a CUDA object file       |
   +---------------+-------+---------------------------------------+
   | IMG_Fatbinary | 0x04  | The image is a CUDA fatbinary file    |
   +---------------+-------+---------------------------------------+
   | IMG_PTX       | 0x05  | The image is a CUDA PTX file          |
   +---------------+-------+---------------------------------------+

.. table:: Offload Kind
   :name: table-offload_kind

   +------------+-------+---------------------------------------+
   |      Name  | Value | Description                           |
   +============+=======+=======================================+
   | OFK_None   | 0x00  | No offloading information provided    |
   +------------+-------+---------------------------------------+
   | OFK_OpenMP | 0x01  | The producer was OpenMP offloading    |
   +------------+-------+---------------------------------------+
   | OFK_CUDA   | 0x02  | The producer was CUDA                 |
   +------------+-------+---------------------------------------+
   | OFK_HIP    | 0x03  | The producer was HIP                  |
   +------------+-------+---------------------------------------+
   | OFK_SYCL   | 0x04  | The producer was SYCL                 |
   +------------+-------+---------------------------------------+

SEE ALSO
--------

:manpage:`clang(1)`, :manpage:`llvm-objdump(1)`
