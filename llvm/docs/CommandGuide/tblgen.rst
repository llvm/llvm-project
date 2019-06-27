tblgen - Target Description To C++ Code Generator
=================================================

.. program:: tblgen

SYNOPSIS
--------

:program:`tblgen` [*options*] [*filename*]

DESCRIPTION
-----------

:program:`tblgen` translates from target description (``.td``) files into C++
code that can be included in the definition of an LLVM target library.  Most
users of LLVM will not need to use this program.  It is only for assisting with
writing an LLVM target backend.

The input and output of :program:`tblgen` is beyond the scope of this short
introduction; please see the :doc:`introduction to TableGen
<../TableGen/index>`.

The *filename* argument specifies the name of a Target Description (``.td``)
file to read as input.

OPTIONS
-------

.. program:: tblgen

.. option:: -help

 Print a summary of command line options.

.. option:: -o filename

 Specify the output file name.  If ``filename`` is ``-``, then
 :program:`tblgen` sends its output to standard output.

.. option:: -I directory

 Specify where to find other target description files for inclusion.  The
 ``directory`` value should be a full or partial path to a directory that
 contains target description files.

.. option:: -asmparsernum N

 Make -gen-asm-parser emit assembly writer number ``N``.

.. option:: -asmwriternum N

 Make -gen-asm-writer emit assembly writer number ``N``.

.. option:: -class className

 Print the enumeration list for this class.

.. option:: -print-records

 Print all records to standard output (default).

.. option:: -dump-json

 Print a JSON representation of all records, suitable for further
 automated processing.

.. option:: -print-enums

 Print enumeration values for a class.

.. option:: -print-sets

 Print expanded sets for testing DAG exprs.

.. option:: -gen-emitter

 Generate machine code emitter.

.. option:: -gen-register-info

 Generate registers and register classes info.

.. option:: -gen-instr-info

 Generate instruction descriptions.

.. option:: -gen-asm-writer

 Generate the assembly writer.

.. option:: -gen-disassembler

 Generate disassembler.

.. option:: -gen-pseudo-lowering

 Generate pseudo instruction lowering.

.. option:: -gen-dag-isel

 Generate a DAG (Directed Acycle Graph) instruction selector.

.. option:: -gen-asm-matcher

 Generate assembly instruction matcher.

.. option:: -gen-dfa-packetizer

 Generate DFA Packetizer for VLIW targets.

.. option:: -gen-fast-isel

 Generate a "fast" instruction selector.

.. option:: -gen-subtarget

 Generate subtarget enumerations.

.. option:: -gen-intrinsic-enums

 Generate intrinsic enums.

.. option:: -gen-intrinsic-impl

 Generate intrinsic implementation.

.. option:: -gen-tgt-intrinsic

 Generate target intrinsic information.

.. option:: -gen-enhanced-disassembly-info

 Generate enhanced disassembly info.

.. option:: -gen-exegesis

 Generate llvm-exegesis tables.

.. option:: -version

 Show the version number of this program.

EXIT STATUS
-----------

If :program:`tblgen` succeeds, it will exit with 0.  Otherwise, if an error
occurs, it will exit with a non-zero value.
