llvm-remarkutil -
==============================================================

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

.. _bitstream2yaml_subcommand:

bitstream2yaml
~~~~~~

.. program:: llvm-remarkutil bitstream2yaml

USAGE: :program:`llvm-remarkutil` bitstream2yaml <input file> -o <output file>

Summary
^^^^^^^^^^^

Takes a bitstream remark file as input, and reserializes that file as YAML.

.. _yaml2bitstream_subcommand:

yaml2bitstream
~~~~~~

.. program:: llvm-remarkutil yaml2bitstream

USAGE: :program:`llvm-remarkutil` yaml2bitstream <input file> -o <output file>

Summary
^^^^^^^^^^^

Takes a YAML remark file as input, and reserializes that file in the bitstream
format.
