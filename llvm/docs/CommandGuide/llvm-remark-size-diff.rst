llvm-remark-size-diff - diff size remarks
=========================================

.. program:: llvm-remark-size-diff

SYNOPSIS
--------

:program:`llvm-remark-size-diff` [*options*] *file_a* *file_b* **--parser** *parser*

DESCRIPTION
-----------

:program:`llvm-remark-size-diff` diffs size
`remarks <https://llvm.org/docs/Remarks.html>`_ in two remark files: ``file_a``
and ``file_b``.

:program:`llvm-remark-size-diff` can be used to gain insight into which
functions were impacted the most by code generation changes.

In most common use-cases ``file_a`` and ``file_b`` will be remarks output by
compiling a **fixed source** with **differing compilers** or
**differing optimization settings**.

:program:`llvm-remark-size-diff` handles both
`YAML <https://llvm.org/docs/Remarks.html#yaml-remarks>`_ and
`bitstream <https://llvm.org/docs/Remarks.html#llvm-bitstream-remarks>`_
remarks.

OPTIONS
-------

.. option:: --parser=<yaml|bitstream>

  Select the type of input remark parser. Required.
  * ``yaml``: The tool will parse YAML remarks.
  * ``bitstream``: The tool will parse bitstream remarks.

.. option:: --report-style=<human|json>

  Output style.
  * ``human``: Human-readable textual report. Default option.
  * ``json``: JSON report.

.. option:: --pretty

  Pretty-print JSON output. Optional.

  If output is not set to JSON, this does nothing.

.. option:: -o=<file>

  Output file for the report. Outputs to stdout by default.

HUMAN-READABLE OUTPUT
---------------------

The human-readable format for :program:`llvm-remark-size-diff` is composed of
two sections:

* Per-function changes.
* A high-level summary of all changes.

Changed Function Section
########################

Suppose you are comparing two remark files OLD and NEW.

For each function with a **changed instruction count** in OLD and NEW,
:program:`llvm-remark-size-diff` will emit a line like below:

::

  (++|--|==) (>|<) function_name, N instrs, M stack B

A breakdown of the format is below:

``(++|--|==)``
  Which of OLD and NEW the ``function_name`` is present in.

  * ``++``: Only in NEW. ("Added")
  * ``--``: Only in OLD. ("Removed")
  * ``==``: In both.

``(>|<)``
  Denotes if ``function_name`` has more instructions or fewer instructions in
  the second file.

  *  ``>``: More instructions in second file than first file.
  *  ``<``: Fewer instructions in second file than in first file.

``function_name``
  The name of the changed function.

``N instrs``
  Second file instruction count - first file instruction count.

``M stack B``
  Second file stack byte count - first file stack byte count.

Summary Section
###############

:program:`llvm-remark-size-diff` will output a high-level summary after
printing all changed functions.

::

  instruction count: N (inst_pct_change%)
  stack byte usage: M (sb_pct_change%)

``N``
  Sum of all instruction count changes between the second and first file.

``inst_pct_change%``
  Percent increase or decrease in instruction count between the second and first
  file.

``M``
  Sum of all stack byte count changes between the second and first file.

``sb_pct_change%``
  Percent increase or decrease in stack byte usage between the second and first
  file.

JSON OUTPUT
-----------

High-Level view
###############

Suppose we are comparing two files, OLD and NEW.

:program:`llvm-remark-size-diff` will output JSON as follows.

::

  "Files": [
    "A": "path/to/OLD",
    "B": "path/to/NEW"
  ]

  "InBoth": [
    ...
  ],

  "OnlyInA": [
    ...
  ],

  "OnlyInB": [
    ...
  ]


``Files``
  Original paths to remark files.

  * ``A``: Path to the first file.
  * ``B``: Path to the second file.

``InBoth``
  Functions present in both files.

``OnlyInA``
  Functions only present in the first file.

``OnlyInB``
  Functions only present in the second file.

Function JSON
#############

The ``InBoth``, ``OnlyInA``, and ``OnlyInB`` sections contain size information
for each function in the input remark files.

::

  {
    "FunctionName" : "function_name"
    "InstCount": [
        INST_COUNT_A,
        INST_COUNT_B
      ],
    "StackSize": [
        STACK_BYTES_A,
        STACK_BYTES_B
      ],
  }

``FunctionName``
  Name of the function.

``InstCount``
  Instruction counts for the function.

  * ``INST_COUNT_A``: Instruction count in OLD.
  * ``INST_COUNT_B``: Instruction count in NEW.

``StackSize``
  Stack byte counts for the function.

  * ``STACK_BYTES_A``: Stack bytes in OLD.
  *  ``STACK_BYTES_B``: Stack bytes in NEW.

Computing Diffs From Function JSON
**********************************

Function JSON does not contain the diffs. Tools consuming JSON output from
:program:`llvm-remark-size-diff` are responsible for computing the diffs
separately.

**To compute the diffs:**

* Instruction count diff: ``INST_COUNT_B - INST_COUNT_A``
* Stack byte count diff: ``STACK_BYTES_B - STACK_BYTES_A``

EXIT STATUS
-----------

:program:`llvm-remark-size-diff` returns 0 on success, and a non-zero value
otherwise.
