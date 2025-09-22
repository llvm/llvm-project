=======
Remarks
=======

.. contents::
   :local:

Introduction to the LLVM remark diagnostics
===========================================

LLVM is able to emit diagnostics from passes describing whether an optimization
has been performed or missed for a particular reason, which should give more
insight to users about what the compiler did during the compilation pipeline.

There are three main remark types:

``Passed``

    Remarks that describe a successful optimization performed by the compiler.

    :Example:

    ::

        foo inlined into bar with (cost=always): always inline attribute

``Missed``

    Remarks that describe an attempt to an optimization by the compiler that
    could not be performed.

    :Example:

    ::

        foo not inlined into bar because it should never be inlined
        (cost=never): noinline function attribute

``Analysis``

    Remarks that describe the result of an analysis, that can bring more
    information to the user regarding the generated code.

    :Example:

    ::

        16 stack bytes in function

    ::

        10 instructions in function

Enabling optimization remarks
=============================

There are two modes that are supported for enabling optimization remarks in
LLVM: through remark diagnostics, or through serialized remarks.

See also the clang flags
`-Rpass <https://clang.llvm.org/docs/UsersManual.html#options-to-emit-optimization-reports>`_
and
`-fsave-optimization-record <http://clang.llvm.org/docs/UsersManual.html#cmdoption-f-no-save-optimization-record>`_.

Remark diagnostics
------------------

Optimization remarks can be emitted as diagnostics. These diagnostics will be
propagated to front-ends if desired, or emitted by tools like :doc:`llc
<CommandGuide/llc>` or :doc:`opt <CommandGuide/opt>`.

.. option:: -pass-remarks=<regex>

  Enables optimization remarks from passes whose name match the given (POSIX)
  regular expression.

.. option:: -pass-remarks-missed=<regex>

  Enables missed optimization remarks from passes whose name match the given
  (POSIX) regular expression.

.. option:: -pass-remarks-analysis=<regex>

  Enables optimization analysis remarks from passes whose name match the given
  (POSIX) regular expression.

Serialized remarks
------------------

While diagnostics are useful during development, it is often more useful to
refer to optimization remarks post-compilation, typically during performance
analysis.

For that, LLVM can serialize the remarks produced for each compilation unit to
a file that can be consumed later.

By default, the format of the serialized remarks is :ref:`YAML
<yamlremarks>`, and it can be accompanied by a :ref:`section <remarkssection>`
in the object files to easily retrieve it.

:doc:`llc <CommandGuide/llc>` and :doc:`opt <CommandGuide/opt>` support the
following options:


``Basic options``

    .. option:: -pass-remarks-output=<filename>

      Enables the serialization of remarks to a file specified in <filename>.

      By default, the output is serialized to :ref:`YAML <yamlremarks>`.

    .. option:: -pass-remarks-format=<format>

      Specifies the output format of the serialized remarks.

      Supported formats:

      * :ref:`yaml <yamlremarks>` (default)
      * :ref:`bitstream <bitstreamremarks>`

``Content configuration``

    .. option:: -pass-remarks-filter=<regex>

      Only passes whose name match the given (POSIX) regular expression will be
      serialized to the final output.

    .. option:: -pass-remarks-with-hotness

      With PGO, include profile count in optimization remarks.

    .. option:: -pass-remarks-hotness-threshold

      The minimum profile count required for an optimization remark to be
      emitted.

Other tools that support remarks:

:program:`llvm-lto`

    .. option:: -lto-pass-remarks-output=<filename>
    .. option:: -lto-pass-remarks-filter=<regex>
    .. option:: -lto-pass-remarks-format=<format>
    .. option:: -lto-pass-remarks-with-hotness
    .. option:: -lto-pass-remarks-hotness-threshold

:program:`gold-plugin` and :program:`lld`

    .. option:: -opt-remarks-filename=<filename>
    .. option:: -opt-remarks-filter=<regex>
    .. option:: -opt-remarks-format=<format>
    .. option:: -opt-remarks-with-hotness

.. _yamlremarks:

YAML remarks
============

A typical remark serialized to YAML looks like this:

.. code-block:: yaml

    --- !<TYPE>
    Pass: <pass>
    Name: <name>
    DebugLoc: { File: <file>, Line: <line>, Column: <column> }
    Function: <function>
    Hotness: <hotness>
    Args:
      - <key>: <value>
        DebugLoc: { File: <arg-file>, Line: <arg-line>, Column: <arg-column> }

The following entries are mandatory:

* ``<TYPE>``: can be ``Passed``, ``Missed``, ``Analysis``,
  ``AnalysisFPCommute``, ``AnalysisAliasing``, ``Failure``.
* ``<pass>``: the name of the pass that emitted this remark.
* ``<name>``: the name of the remark coming from ``<pass>``.
* ``<function>``: the mangled name of the function.

If a ``DebugLoc`` entry is specified, the following fields are required:

* ``<file>``
* ``<line>``
* ``<column>``

If an ``arg`` entry is specified, the following fields are required:

* ``<key>``
* ``<value>``

If a ``DebugLoc`` entry is specified within an ``arg`` entry, the following
fields are required:

* ``<arg-file>``
* ``<arg-line>``
* ``<arg-column>``

.. _optviewer:

YAML metadata
-------------

The metadata used together with the YAML format is:

* a magic number: "REMARKS\\0"
* the version number: a little-endian uint64_t
* 8 zero bytes. This space was previously used to encode the size of a string
  table. String table support for YAML remarks has been removed, use the
  bitstream format instead.

Optional:

* the absolute file path to the serialized remark diagnostics: a
  null-terminated string.

When the metadata is serialized separately from the remarks, the file path
should be present and point to the file where the remarks are serialized to.

In case the metadata only acts as a header to the remarks, the file path can be
omitted.

.. _bitstreamremarks:

LLVM bitstream remarks
======================

This format is using :doc:`LLVM bitstream <BitCodeFormat>` to serialize remarks
and their associated metadata.

A bitstream remark stream can be identified by the magic number ``"RMRK"`` that
is placed at the very beginning.

The format for serializing remarks is composed of two different block types:

.. _bitstreamremarksmetablock:

META_BLOCK
----------

The block providing information about the rest of the content in the stream.

Exactly one block is expected. Having multiple metadata blocks is an error.

This block can contain the following records:

.. _bitstreamremarksrecordmetacontainerinfo:

``RECORD_META_CONTAINER_INFO``

    The container version and type.

    Version: u32

    Type:    u2

.. _bitstreamremarksrecordmetaremarkversion:

``RECORD_META_REMARK_VERSION``

    The version of the remark entries. This can change independently from the
    container version.

    Version: u32

.. _bitstreamremarksrecordmetastrtab:

``RECORD_META_STRTAB``

    The string table used by the remark entries. The format of the string table
    is a sequence of strings separated by ``\0``.

.. _bitstreamremarksrecordmetaexternalfile:

``RECORD_META_EXTERNAL_FILE``

    The external remark file path that contains the remark blocks associated
    with this metadata. This is an absolute path.

.. _bitstreamremarksremarkblock:

REMARK_BLOCK
------------

The block describing a remark entry.

0 or more blocks per file are allowed. Each block will depend on the
:ref:`META_BLOCK <bitstreamremarksmetablock>` in order to be parsed correctly.

This block can contain the following records:

``RECORD_REMARK_HEADER``

    The header of the remark. This contains all the mandatory information about
    a remark.

    +---------------+---------------------------+
    | Type          | u3                        |
    +---------------+---------------------------+
    | Remark name   | VBR6 (string table index) |
    +---------------+---------------------------+
    | Pass name     | VBR6 (string table index) |
    +---------------+---------------------------+
    | Function name | VBR6 (string table index) |
    +---------------+---------------------------+

``RECORD_REMARK_DEBUG_LOC``

    The source location for the corresponding remark. This record is optional.

    +--------+---------------------------+
    | File   | VBR7 (string table index) |
    +--------+---------------------------+
    | Line   | u32                       |
    +--------+---------------------------+
    | Column | u32                       |
    +--------+---------------------------+

``RECORD_REMARK_HOTNESS``

    The hotness of the remark. This record is optional.

    +---------------+---------------------+
    | Hotness | VBR8 (string table index) |
    +---------------+---------------------+

``RECORD_REMARK_ARG_WITH_DEBUGLOC``

    A remark argument with an associated debug location.

    +--------+---------------------------+
    | Key    | VBR7 (string table index) |
    +--------+---------------------------+
    | Value  | VBR7 (string table index) |
    +--------+---------------------------+
    | File   | VBR7 (string table index) |
    +--------+---------------------------+
    | Line   | u32                       |
    +--------+---------------------------+
    | Column | u32                       |
    +--------+---------------------------+

``RECORD_REMARK_ARG_WITHOUT_DEBUGLOC``

    A remark argument with an associated debug location.

    +--------+---------------------------+
    | Key    | VBR7 (string table index) |
    +--------+---------------------------+
    | Value  | VBR7 (string table index) |
    +--------+---------------------------+

The remark container
--------------------

The bitstream remark container supports multiple types:

.. _bitstreamremarksfileexternal:

``RemarksFileExternal: a link to an external remarks file``

    This container type expects only a :ref:`META_BLOCK <bitstreamremarksmetablock>` containing only:

    * :ref:`RECORD_META_CONTAINER_INFO <bitstreamremarksrecordmetacontainerinfo>`
    * :ref:`RECORD_META_STRTAB <bitstreamremarksrecordmetastrtab>`
    * :ref:`RECORD_META_EXTERNAL_FILE <bitstreamremarksrecordmetaexternalfile>`

    Typically, this is emitted in a section in the object files, allowing
    clients to retrieve remarks and their associated metadata directly from
    intermediate products.

    The container versions of the external separate container should match in order to
    have a well-formed file.

.. _bitstreamremarksfile:

``RemarksFile: a standalone remarks file``

    This container type expects a :ref:`META_BLOCK <bitstreamremarksmetablock>` containing only:

    * :ref:`RECORD_META_CONTAINER_INFO <bitstreamremarksrecordmetacontainerinfo>`
    * :ref:`RECORD_META_REMARK_VERSION <bitstreamremarksrecordmetaremarkversion>`

    Then, this container type expects 1 or more :ref:`REMARK_BLOCK <bitstreamremarksremarkblock>`.
    If no remarks are emitted, the meta blocks are also not emitted, so the file is empty.

    After the remark blocks, another :ref:`META_BLOCK <bitstreamremarksmetablock>` is emitted, containing:
    * :ref:`RECORD_META_STRTAB <bitstreamremarksrecordmetastrtab>`

    When the parser reads this container type, it jumps to the end of the file
    to read the string table before parsing the individual remarks.

    Standalone remarks files can be referenced by the
    :ref:`RECORD_META_EXTERNAL_FILE <bitstreamremarksrecordmetaexternalfile>`
    entry in the :ref:`RemarksFileExternal
    <bitstreamremarksfileexternal>` container.

.. FIXME: Add complete output of :program:`llvm-bcanalyzer` on the different container types (once format changes are completed)

opt-viewer
==========

The ``opt-viewer`` directory contains a collection of tools that visualize and
summarize serialized remarks.

The tools only support the ``yaml`` format.

.. _optviewerpy:

opt-viewer.py
-------------

Output a HTML page which gives visual feedback on compiler interactions with
your program.

    :Examples:

    ::

        $ opt-viewer.py my_yaml_file.opt.yaml

    ::

        $ opt-viewer.py my_build_dir/


opt-stats.py
------------

Output statistics about the optimization remarks in the input set.

    :Example:

    ::

        $ opt-stats.py my_yaml_file.opt.yaml

        Total number of remarks           3


        Top 10 remarks by pass:
          inline                         33%
          asm-printer                    33%
          prologepilog                   33%

        Top 10 remarks:
          asm-printer/InstructionCount   33%
          inline/NoDefinition            33%
          prologepilog/StackSize         33%

opt-diff.py
-----------

Produce a new YAML file which contains all of the changes in optimizations
between two YAML files.

Typically, this tool should be used to do diffs between:

* new compiler + fixed source vs old compiler + fixed source
* fixed compiler + new source vs fixed compiler + old source

This diff file can be displayed using :ref:`opt-viewer.py <optviewerpy>`.

    :Example:

    ::

        $ opt-diff.py my_opt_yaml1.opt.yaml my_opt_yaml2.opt.yaml -o my_opt_diff.opt.yaml
        $ opt-viewer.py my_opt_diff.opt.yaml

.. _remarkssection:

Emitting remark diagnostics in the object file
==============================================

A section containing metadata on remark diagnostics will be emitted for the
following formats:

* ``bitstream``

This can be overridden by using the flag ``-remarks-section=<bool>``.

The section is named:

* ``__LLVM,__remarks`` (MachO)

C API
=====

LLVM provides a library that can be used to parse remarks through a shared
library named ``libRemarks``.

The typical usage through the C API is like the following:

.. code-block:: c

    LLVMRemarkParserRef Parser = LLVMRemarkParserCreateYAML(Buf, Size);
    LLVMRemarkEntryRef Remark = NULL;
    while ((Remark = LLVMRemarkParserGetNext(Parser))) {
       // use Remark
       LLVMRemarkEntryDispose(Remark); // Release memory.
    }
    bool HasError = LLVMRemarkParserHasError(Parser);
    LLVMRemarkParserDispose(Parser);

Remark streamers
================

The ``RemarkStreamer`` interface is used to unify the serialization
capabilities of remarks across all the components that can generate remarks.

All remark serialization should go through the main remark streamer, the
``llvm::remarks::RemarkStreamer`` set up in the ``LLVMContext``. The interface
takes remark objects converted to ``llvm::remarks::Remark``, and takes care of
serializing it to the requested format, using the requested type of metadata,
etc.

Typically, a specialized remark streamer will hold a reference to the one set
up in the ``LLVMContext``, and will operate on its own type of diagnostics.

For example, LLVM IR passes will emit ``llvm::DiagnosticInfoOptimization*``
that get converted to ``llvm::remarks::Remark`` objects.  Then, clang could set
up its own specialized remark streamer that takes ``clang::Diagnostic``
objects. This can allow various components of the frontend to emit remarks
using the same techniques as the LLVM remarks.

This gives us the following advantages:

* Composition: during the compilation pipeline, multiple components can set up
  their specialized remark streamers that all emit remarks through the same
  main streamer.
* Re-using the remark infrastructure in ``lib/Remarks``.
* Using the same file and format for the remark emitters created throughout the
  compilation.

at the cost of an extra layer of abstraction.

.. FIXME: add documentation for llvm-opt-report.
.. FIXME: add documentation for Passes supporting optimization remarks
.. FIXME: add documentation for IR Passes
.. FIXME: add documentation for CodeGen Passes
