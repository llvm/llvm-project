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

opt-viewer
==========

The ``opt-viewer`` directory contains a collection of tools that visualize and
summarize serialized remarks.

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

A section containing metadata on remark diagnostics will be emitted when
-remarks-section is passed. The section contains:

* a magic number: "REMARKS\\0"
* the version number: a little-endian uint64_t
* the total size of the string table (the size itself excluded):
  little-endian uint64_t
* a list of null-terminated strings
* the absolute file path to the serialized remark diagnostics: a
  null-terminated string.

The section is named:

* ``__LLVM,__remarks`` (MachO)
* ``.remarks`` (ELF)

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

.. FIXME: add documentation for llvm-opt-report.
.. FIXME: add documentation for Passes supporting optimization remarks
.. FIXME: add documentation for IR Passes
.. FIXME: add documentation for CodeGen Passes
