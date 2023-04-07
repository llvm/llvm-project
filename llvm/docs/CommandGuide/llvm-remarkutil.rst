llvm-remarkutil - Remark utility
================================

.. program:: llvm-remarkutil

SYNOPSIS
--------

:program:`llvm-remarkutil` [*subcommmand*] [*options*]

DESCRIPTION
-----------

Utility for displaying information from, and converting between different
`remark <https://llvm.org/docs/Remarks.html>`_ formats.

Subcommands
-----------

  * :ref:`bitstream2yaml_subcommand` - Reserialize bitstream remarks to YAML.
  * :ref:`yaml2bitstream_subcommand` - Reserialize YAML remarks to bitstream.
  * :ref:`instruction-count_subcommand` - Output function instruction counts.

.. _bitstream2yaml_subcommand:

bitstream2yaml
~~~~~~~~~~~~~~

.. program:: llvm-remarkutil bitstream2yaml

USAGE: :program:`llvm-remarkutil` bitstream2yaml <input file> -o <output file>

Summary
^^^^^^^

Takes a bitstream remark file as input, and reserializes that file as YAML.

.. _yaml2bitstream_subcommand:

yaml2bitstream
~~~~~~~~~~~~~~

.. program:: llvm-remarkutil yaml2bitstream

USAGE: :program:`llvm-remarkutil` yaml2bitstream <input file> -o <output file>

Summary
^^^^^^^

Takes a YAML remark file as input, and reserializes that file in the bitstream
format.

.. _instruction-count_subcommand:

instruction-count
~~~~~~~~~~~~~~~~~

.. program:: llvm-remarkutil instruction-count

USAGE: :program:`llvm-remarkutil` instruction-count <input file> --parser=<bitstream|yaml> -o <output file>

Summary
^^^^^^^

Outputs instruction count remarks for every function. Instruction count remarks
encode the number of instructions in a function at assembly printing time.

Instruction count remarks require asm-printer remarks.

CSV format is as follows:

::
  Function,InstructionCount
  foo,123
