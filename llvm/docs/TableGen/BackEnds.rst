=================
TableGen BackEnds
=================

.. contents::
   :local:

Introduction
============

TableGen backends are at the core of TableGen's functionality. The source files
provide the semantics to a generated (in memory) structure, but it's up to the
backend to print this out in a way that is meaningful to the user (normally a
C program including a file or a textual list of warnings, options and error
messages).

TableGen is used by both LLVM and Clang with very different goals. LLVM uses it
as a way to automate the generation of massive amounts of information regarding
instructions, schedules, cores and architecture features. Some backends generate
output that is consumed by more than one source file, so they need to be created
in a way that is easy to use pre-processor tricks. Some backends can also print
C code structures, so that they can be directly included as-is.

Clang, on the other hand, uses it mainly for diagnostic messages (errors,
warnings, tips) and attributes, so more on the textual end of the scale.

LLVM BackEnds
=============

.. warning::
   This document is raw. Each section below needs three sub-sections: description
   of its purpose with a list of users, output generated from generic input, and
   finally why it needed a new backend (in case there's something similar).

Overall, each backend will take the same TableGen file type and transform into
similar output for different targets/uses. There is an implicit contract between
the TableGen files, the back-ends and their users.

For instance, a global contract is that each back-end produces macro-guarded
sections. Based on whether the file is included by a header or a source file,
or even in which context of each file the include is being used, you have
todefine a macro just before including it, to get the right output:

.. code-block:: c++

  #define GET_REGINFO_TARGET_DESC
  #include "ARMGenRegisterInfo.inc"

And just part of the generated file would be included. This is useful if
you need the same information in multiple formats (instantiation, initialization,
getter/setter functions, etc) from the same source TableGen file without having
to re-compile the TableGen file multiple times.

Sometimes, multiple macros might be defined before the same include file to
output multiple blocks:

.. code-block:: c++

  #define GET_REGISTER_MATCHER
  #define GET_SUBTARGET_FEATURE_NAME
  #define GET_MATCHER_IMPLEMENTATION
  #include "ARMGenAsmMatcher.inc"

The macros will be undef'd automatically as they're used, in the include file.

On all LLVM back-ends, the ``llvm-tblgen`` binary will be executed on the root
TableGen file ``<Target>.td``, which should include all others. This guarantees
that all information needed is accessible, and that no duplication is needed
in the TableGen files.

CodeEmitter
-----------

**Purpose**: CodeEmitterGen uses the descriptions of instructions and their fields to
construct an automated code emitter: a function that, given a MachineInstr,
returns the (currently, 32-bit unsigned) value of the instruction.

**Output**: C++ code, implementing the target's CodeEmitter
class by overriding the virtual functions as ``<Target>CodeEmitter::function()``.

**Usage**: Used to include directly at the end of ``<Target>MCCodeEmitter.cpp``.

RegisterInfo
------------

**Purpose**: This tablegen backend is responsible for emitting a description of a target
register file for a code generator.  It uses instances of the Register,
RegisterAliases, and RegisterClass classes to gather this information.

**Output**: C++ code with enums and structures representing the register mappings,
properties, masks, etc.

**Usage**: Both on ``<Target>BaseRegisterInfo`` and ``<Target>MCTargetDesc`` (headers
and source files) with macros defining in which they are for declaration vs.
initialization issues.

InstrInfo
---------

**Purpose**: This tablegen backend is responsible for emitting a description of the target
instruction set for the code generator. (what are the differences from CodeEmitter?)

**Output**: C++ code with enums and structures representing the instruction mappings,
properties, masks, etc.

**Usage**: Both on ``<Target>BaseInstrInfo`` and ``<Target>MCTargetDesc`` (headers
and source files) with macros defining in which they are for declaration vs.
initialization issues.

AsmWriter
---------

**Purpose**: Emits an assembly printer for the current target.

**Output**: Implementation of ``<Target>InstPrinter::printInstruction()``, among
other things.

**Usage**: Included directly into ``InstPrinter/<Target>InstPrinter.cpp``.

AsmMatcher
----------

**Purpose**: Emits a target specifier matcher for
converting parsed assembly operands in the MCInst structures. It also
emits a matcher for custom operand parsing. Extensive documentation is
written on the ``AsmMatcherEmitter.cpp`` file.

**Output**: Assembler parsers' matcher functions, declarations, etc.

**Usage**: Used in back-ends' ``AsmParser/<Target>AsmParser.cpp`` for
building the AsmParser class.

Disassembler
------------

**Purpose**: Contains disassembler table emitters for various
architectures. Extensive documentation is written on the
``DisassemblerEmitter.cpp`` file.

**Output**: Decoding tables, static decoding functions, etc.

**Usage**: Directly included in ``Disassembler/<Target>Disassembler.cpp``
to cater for all default decodings, after all hand-made ones.

PseudoLowering
--------------

**Purpose**: Generate pseudo instruction lowering.

**Output**: Implements ``<Target>AsmPrinter::emitPseudoExpansionLowering()``.

**Usage**: Included directly into ``<Target>AsmPrinter.cpp``.

CallingConv
-----------

**Purpose**: Responsible for emitting descriptions of the calling
conventions supported by this target.

**Output**: Implement static functions to deal with calling conventions
chained by matching styles, returning false on no match.

**Usage**: Used in ISelLowering and FastIsel as function pointers to
implementation returned by a CC selection function.

DAGISel
-------

**Purpose**: Generate a DAG instruction selector.

**Output**: Creates huge functions for automating DAG selection.

**Usage**: Included in ``<Target>ISelDAGToDAG.cpp`` inside the target's
implementation of ``SelectionDAGISel``.

DFAPacketizer
-------------

**Purpose**: This class parses the Schedule.td file and produces an API that
can be used to reason about whether an instruction can be added to a packet
on a VLIW architecture. The class internally generates a deterministic finite
automaton (DFA) that models all possible mappings of machine instructions
to functional units as instructions are added to a packet.

**Output**: Scheduling tables for GPU back-ends (Hexagon, AMD).

**Usage**: Included directly on ``<Target>InstrInfo.cpp``.

FastISel
--------

**Purpose**: This tablegen backend emits code for use by the "fast"
instruction selection algorithm. See the comments at the top of
lib/CodeGen/SelectionDAG/FastISel.cpp for background. This file
scans through the target's tablegen instruction-info files
and extracts instructions with obvious-looking patterns, and it emits
code to look up these instructions by type and operator.

**Output**: Generates ``Predicate`` and ``FastEmit`` methods.

**Usage**: Implements private methods of the targets' implementation
of ``FastISel`` class.

Subtarget
---------

**Purpose**: Generate subtarget enumerations.

**Output**: Enums, globals, local tables for sub-target information.

**Usage**: Populates ``<Target>Subtarget`` and
``MCTargetDesc/<Target>MCTargetDesc`` files (both headers and source).

Intrinsic
---------

**Purpose**: Generate (target) intrinsic information.

OptParserDefs
-------------

**Purpose**: Print enum values for a class.

SearchableTables
----------------

**Purpose**: Generate custom searchable tables.

**Output**: Enums, global tables and lookup helper functions.

**Usage**: This backend allows generating free-form, target-specific tables
from TableGen records. The ARM and AArch64 targets use this backend to generate
tables of system registers; the AMDGPU target uses it to generate meta-data
about complex image and memory buffer instructions.

More documentation is available in ``include/llvm/TableGen/SearchableTable.td``,
which also contains the definitions of TableGen classes which must be
instantiated in order to define the enums and tables emitted by this backend.

CTags
-----

**Purpose**: This tablegen backend emits an index of definitions in ctags(1)
format. A helper script, utils/TableGen/tdtags, provides an easier-to-use
interface; run 'tdtags -H' for documentation.

X86EVEX2VEX
-----------

**Purpose**: This X86 specific tablegen backend emits tables that map EVEX
encoded instructions to their VEX encoded identical instruction.

Clang BackEnds
==============

ClangAttrClasses
----------------

**Purpose**: Creates Attrs.inc, which contains semantic attribute class
declarations for any attribute in ``Attr.td`` that has not set ``ASTNode = 0``.
This file is included as part of ``Attr.h``.

ClangAttrParserStringSwitches
-----------------------------

**Purpose**: Creates AttrParserStringSwitches.inc, which contains
StringSwitch::Case statements for parser-related string switches. Each switch
is given its own macro (such as ``CLANG_ATTR_ARG_CONTEXT_LIST``, or
``CLANG_ATTR_IDENTIFIER_ARG_LIST``), which is expected to be defined before
including AttrParserStringSwitches.inc, and undefined after.

ClangAttrImpl
-------------

**Purpose**: Creates AttrImpl.inc, which contains semantic attribute class
definitions for any attribute in ``Attr.td`` that has not set ``ASTNode = 0``.
This file is included as part of ``AttrImpl.cpp``.

ClangAttrList
-------------

**Purpose**: Creates AttrList.inc, which is used when a list of semantic
attribute identifiers is required. For instance, ``AttrKinds.h`` includes this
file to generate the list of ``attr::Kind`` enumeration values. This list is
separated out into multiple categories: attributes, inheritable attributes, and
inheritable parameter attributes. This categorization happens automatically
based on information in ``Attr.td`` and is used to implement the ``classof``
functionality required for ``dyn_cast`` and similar APIs.

ClangAttrPCHRead
----------------

**Purpose**: Creates AttrPCHRead.inc, which is used to deserialize attributes
in the ``ASTReader::ReadAttributes`` function.

ClangAttrPCHWrite
-----------------

**Purpose**: Creates AttrPCHWrite.inc, which is used to serialize attributes in
the ``ASTWriter::WriteAttributes`` function.

ClangAttrSpellings
---------------------

**Purpose**: Creates AttrSpellings.inc, which is used to implement the
``__has_attribute`` feature test macro.

ClangAttrSpellingListIndex
--------------------------

**Purpose**: Creates AttrSpellingListIndex.inc, which is used to map parsed
attribute spellings (including which syntax or scope was used) to an attribute
spelling list index. These spelling list index values are internal
implementation details exposed via
``AttributeList::getAttributeSpellingListIndex``.

ClangAttrVisitor
-------------------

**Purpose**: Creates AttrVisitor.inc, which is used when implementing 
recursive AST visitors.

ClangAttrTemplateInstantiate
----------------------------

**Purpose**: Creates AttrTemplateInstantiate.inc, which implements the
``instantiateTemplateAttribute`` function, used when instantiating a template
that requires an attribute to be cloned.

ClangAttrParsedAttrList
-----------------------

**Purpose**: Creates AttrParsedAttrList.inc, which is used to generate the
``AttributeList::Kind`` parsed attribute enumeration.

ClangAttrParsedAttrImpl
-----------------------

**Purpose**: Creates AttrParsedAttrImpl.inc, which is used by
``AttributeList.cpp`` to implement several functions on the ``AttributeList``
class. This functionality is implemented via the ``AttrInfoMap ParsedAttrInfo``
array, which contains one element per parsed attribute object.

ClangAttrParsedAttrKinds
------------------------

**Purpose**: Creates AttrParsedAttrKinds.inc, which is used to implement the
``AttributeList::getKind`` function, mapping a string (and syntax) to a parsed
attribute ``AttributeList::Kind`` enumeration.

ClangAttrDump
-------------

**Purpose**: Creates AttrDump.inc, which dumps information about an attribute.
It is used to implement ``ASTDumper::dumpAttr``.

ClangDiagsDefs
--------------

Generate Clang diagnostics definitions.

ClangDiagGroups
---------------

Generate Clang diagnostic groups.

ClangDiagsIndexName
-------------------

Generate Clang diagnostic name index.

ClangCommentNodes
-----------------

Generate Clang AST comment nodes.

ClangDeclNodes
--------------

Generate Clang AST declaration nodes.

ClangStmtNodes
--------------

Generate Clang AST statement nodes.

ClangSACheckers
---------------

Generate Clang Static Analyzer checkers.

ClangCommentHTMLTags
--------------------

Generate efficient matchers for HTML tag names that are used in documentation comments.

ClangCommentHTMLTagsProperties
------------------------------

Generate efficient matchers for HTML tag properties.

ClangCommentHTMLNamedCharacterReferences
----------------------------------------

Generate function to translate named character references to UTF-8 sequences.

ClangCommentCommandInfo
-----------------------

Generate command properties for commands that are used in documentation comments.

ClangCommentCommandList
-----------------------

Generate list of commands that are used in documentation comments.

ArmNeon
-------

Generate arm_neon.h for clang.

ArmNeonSema
-----------

Generate ARM NEON sema support for clang.

ArmNeonTest
-----------

Generate ARM NEON tests for clang.

AttrDocs
--------

**Purpose**: Creates ``AttributeReference.rst`` from ``AttrDocs.td``, and is
used for documenting user-facing attributes.

General BackEnds
================

JSON
----

**Purpose**: Output all the values in every ``def``, as a JSON data
structure that can be easily parsed by a variety of languages. Useful
for writing custom backends without having to modify TableGen itself,
or for performing auxiliary analysis on the same TableGen data passed
to a built-in backend.

**Output**:

The root of the output file is a JSON object (i.e. dictionary),
containing the following fixed keys:

* ``!tablegen_json_version``: a numeric version field that will
  increase if an incompatible change is ever made to the structure of
  this data. The format described here corresponds to version 1.

* ``!instanceof``: a dictionary whose keys are the class names defined
  in the TableGen input. For each key, the corresponding value is an
  array of strings giving the names of ``def`` records that derive
  from that class. So ``root["!instanceof"]["Instruction"]``, for
  example, would list the names of all the records deriving from the
  class ``Instruction``.

For each ``def`` record, the root object also has a key for the record
name. The corresponding value is a subsidiary object containing the
following fixed keys:

* ``!superclasses``: an array of strings giving the names of all the
  classes that this record derives from.

* ``!fields``: an array of strings giving the names of all the variables
  in this record that were defined with the ``field`` keyword.

* ``!name``: a string giving the name of the record. This is always
  identical to the key in the JSON root object corresponding to this
  record's dictionary. (If the record is anonymous, the name is
  arbitrary.)

* ``!anonymous``: a boolean indicating whether the record's name was
  specified by the TableGen input (if it is ``false``), or invented by
  TableGen itself (if ``true``).

For each variable defined in a record, the ``def`` object for that
record also has a key for the variable name. The corresponding value
is a translation into JSON of the variable's value, using the
conventions described below.

Some TableGen data types are translated directly into the
corresponding JSON type:

* A completely undefined value (e.g. for a variable declared without
  initializer in some superclass of this record, and never initialized
  by the record itself or any other superclass) is emitted as the JSON
  ``null`` value.

* ``int`` and ``bit`` values are emitted as numbers. Note that
  TableGen ``int`` values are capable of holding integers too large to
  be exactly representable in IEEE double precision. The integer
  literal in the JSON output will show the full exact integer value.
  So if you need to retrieve large integers with full precision, you
  should use a JSON reader capable of translating such literals back
  into 64-bit integers without losing precision, such as Python's
  standard ``json`` module.

* ``string`` and ``code`` values are emitted as JSON strings.

* ``list<T>`` values, for any element type ``T``, are emitted as JSON
  arrays. Each element of the array is represented in turn using these
  same conventions.

* ``bits`` values are also emitted as arrays. A ``bits`` array is
  ordered from least-significant bit to most-significant. So the
  element with index ``i`` corresponds to the bit described as
  ``x{i}`` in TableGen source. However, note that this means that
  scripting languages are likely to *display* the array in the
  opposite order from the way it appears in the TableGen source or in
  the diagnostic ``-print-records`` output.

All other TableGen value types are emitted as a JSON object,
containing two standard fields: ``kind`` is a discriminator describing
which kind of value the object represents, and ``printable`` is a
string giving the same representation of the value that would appear
in ``-print-records``.

* A reference to a ``def`` object has ``kind=="def"``, and has an
  extra field ``def`` giving the name of the object referred to.

* A reference to another variable in the same record has
  ``kind=="var"``, and has an extra field ``var`` giving the name of
  the variable referred to.

* A reference to a specific bit of a ``bits``-typed variable in the
  same record has ``kind=="varbit"``, and has two extra fields:
  ``var`` gives the name of the variable referred to, and ``index``
  gives the index of the bit.

* A value of type ``dag`` has ``kind=="dag"``, and has two extra
  fields. ``operator`` gives the initial value after the opening
  parenthesis of the dag initializer; ``args`` is an array giving the
  following arguments. The elements of ``args`` are arrays of length
  2, giving the value of each argument followed by its colon-suffixed
  name (if any). For example, in the JSON representation of the dag
  value ``(Op 22, "hello":$foo)`` (assuming that ``Op`` is the name of
  a record defined elsewhere with a ``def`` statement):

  * ``operator`` will be an object in which ``kind=="def"`` and
    ``def=="Op"``

  * ``args`` will be the array ``[[22, null], ["hello", "foo"]]``.

* If any other kind of value or complicated expression appears in the
  output, it will have ``kind=="complex"``, and no additional fields.
  These values are not expected to be needed by backends. The standard
  ``printable`` field can be used to extract a representation of them
  in TableGen source syntax if necessary.

How to write a back-end
=======================

TODO.

Until we get a step-by-step HowTo for writing TableGen backends, you can at
least grab the boilerplate (build system, new files, etc.) from Clang's
r173931.

TODO: How they work, how to write one.  This section should not contain details
about any particular backend, except maybe ``-print-enums`` as an example.  This
should highlight the APIs in ``TableGen/Record.h``.

