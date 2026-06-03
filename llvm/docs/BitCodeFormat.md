# LLVM Bitcode File Format

```{contents}
:local:

```
## Abstract

This document describes the LLVM bitstream file format and the encoding of the
LLVM IR into it.

## Overview

What is commonly known as the LLVM bitcode file format (also, sometimes
anachronistically known as bytecode) is actually two things: a
{ref}`bitstream container format` and an {ref}`encoding of LLVM IR` into the container format.

The bitstream format is an abstract encoding of structured data, very similar to
XML in some ways.  Like XML, bitstream files contain tags, and nested
structures, and you can parse the file without having to understand the tags.
Unlike XML, the bitstream format is a binary encoding, and unlike XML it
provides a mechanism for the file to self-describe "abbreviations", which are
effectively size optimizations for the content.

LLVM IR files may be optionally embedded into a {ref}`wrapper` structure, or in a
{ref}`native object file`. Both of these mechanisms make it easy to embed extra
data along with LLVM IR files.

This document first describes the LLVM bitstream format, describes the wrapper
format, then describes the record structure used by LLVM IR files.

(bitstream container format)=

## Bitstream Format

The bitstream format is literally a stream of bits, with a very simple
structure.  This structure consists of the following concepts:

* A "{ref}`magic number`" that identifies the contents of the stream.

* Encoding {ref}`primitives` like variable bit-rate integers.

* {ref}`Blocks`, which define nested content.

* {ref}`Data Records`, which describe entities within the file.

* Abbreviations, which specify compression optimizations for the file.

Note that the {doc}`llvm-bcanalyzer <CommandGuide/llvm-bcanalyzer>` tool can be
used to dump and inspect arbitrary bitstreams, which is very useful for
understanding the encoding.

(magic number)=

### Magic Numbers

The first four bytes of a bitstream are used as an application-specific magic
number.  Generic bitcode tools may look at the first four bytes to determine
whether the stream is a known stream type.  However, these tools should *not*
determine whether a bitstream is valid based on its magic number alone.  New
application-specific bitstream formats are being developed all the time; tools
should not reject them just because they have a hitherto unseen magic number.

(primitives)=

### Primitives

A bitstream literally consists of a stream of bits, which are read in order
starting with the least significant bit of each byte.  The stream is made up of
a number of primitive values that encode a stream of unsigned integer values.
These integers are encoded in two ways: either as {ref}`fixed-width value` or as
{ref}`variable-width value`.

(Fixed Width Integers)=
(fixed-width value)=

#### Fixed Width Integers

Fixed-width integer values have their low bits emitted directly to the file.
For example, a 3-bit integer value encodes 1 as 001.  Fixed-width integers are
used when there are a well-known number of options for a field.  For example,
boolean values are usually encoded with a 1-bit-wide integer.

(Variable Width Integers)=
(Variable Width Integer)=
(variable-width value)=

#### Variable Width Integers

Variable-width integer (VBR) values encode values of arbitrary size, optimizing
for the case where the values are small.  Given a 4-bit VBR field, any 3-bit
value (0 through 7) is encoded directly, with the high bit set to zero.  Values
larger than N-1 bits emit their bits in a series of N-1 bit chunks, where all
but the last set the high bit.

For example, the value 30 (0x1E) is encoded as 62 (0b0011'1110) when emitted as
a vbr4 value.  The first set of four bits starting from the least significant
indicates the value 6 (110) with a continuation piece (indicated by a high bit
of 1).  The next set of four bits indicates a value of 24 (011 << 3) with no
continuation.  The sum (6+24) yields the value 30.

(char6-encoded value)=

#### 6-bit characters

6-bit characters encode common characters into a fixed 6-bit field.  They
represent the following characters with the following 6-bit values:

```
'a' .. 'z' ---  0 .. 25
'A' .. 'Z' --- 26 .. 51
'0' .. '9' --- 52 .. 61
       '.' --- 62
       '_' --- 63

```
This encoding is only suitable for encoding characters and strings that consist
only of the above characters.  It is completely incapable of encoding characters
not in the set.

#### Word Alignment

Occasionally, it is useful to emit zero bits until the bitstream is a multiple
of 32 bits.  This ensures that the bit position in the stream can be represented
as a multiple of 32-bit words.

### Abbreviation IDs

A bitstream is a sequential series of {ref}`Blocks` and {ref}`Data Records`.  Both of
these start with an abbreviation ID encoded as a fixed-bitwidth field.  The
width is specified by the current block, as described below.  The value of the
abbreviation ID specifies either a builtin ID (which have special meanings,
defined below) or one of the abbreviation IDs defined for the current block by
the stream itself.

The set of builtin abbrev IDs is:

* 0 - {ref}`END_BLOCK` --- This abbrev ID marks the end of the current block.

* 1 - {ref}`ENTER_SUBBLOCK` --- This abbrev ID marks the beginning of a new
  block.

* 2 - {ref}`DEFINE_ABBREV` --- This defines a new abbreviation.

* 3 - {ref}`UNABBREV_RECORD` --- This ID specifies the definition of an
  unabbreviated record.

Abbreviation IDs 4 and above are defined by the stream itself, and specify an
{ref}`abbreviated record encoding`.

(Blocks)=

### Blocks

Blocks in a bitstream denote nested regions of the stream, and are identified by
a content-specific id number (for example, LLVM IR uses an ID of 12 to represent
function bodies).  Block IDs 0-7 are reserved for {ref}`standard blocks` whose
meaning is defined by Bitcode; block IDs 8 and greater are application
specific. Nested blocks capture the hierarchical structure of the data encoded
in it, and various properties are associated with blocks as the file is parsed.
Block definitions allow the reader to efficiently skip blocks in constant time
if the reader wants a summary of blocks, or if it wants to efficiently skip data
it does not understand.  The LLVM IR reader uses this mechanism to skip function
bodies, lazily reading them on demand.

When reading and encoding the stream, several properties are maintained for the
block.  In particular, each block maintains:

1. A current abbrev id width.  This value starts at 2 at the beginning of the
   stream, and is set every time a block record is entered.  The block entry
   specifies the abbrev id width for the body of the block.

1. A set of abbreviations.  Abbreviations may be defined within a block, in
   which case they are only defined in that block (neither subblocks nor
   enclosing blocks see the abbreviation).  Abbreviations can also be defined
   inside a {ref}`BLOCKINFO` block, in which case they are defined in all blocks
   that match the ID that the `BLOCKINFO` block is describing.

As sub blocks are entered, these properties are saved and the new sub-block has
its own set of abbreviations, and its own abbrev id width.  When a sub-block is
popped, the saved values are restored.

(ENTER_SUBBLOCK)=

#### ENTER_SUBBLOCK Encoding

<code>[ENTER_SUBBLOCK, blockid<sub>vbr8</sub>, newabbrevlen<sub>vbr4</sub>, &lt;align32bits&gt;, blocklen_32]</code>

The `ENTER_SUBBLOCK` abbreviation ID specifies the start of a new block
record.  The `blockid` value is encoded as an 8-bit VBR identifier, and
indicates the type of block being entered, which can be a {ref}`standard block` or
an application-specific block.  The `newabbrevlen` value is a 4-bit VBR, which
specifies the abbrev id width for the sub-block.  The `blocklen` value is a
32-bit aligned value that specifies the size of the subblock in 32-bit
words. This value allows the reader to skip over the entire block in one jump.

(END_BLOCK)=

#### END_BLOCK Encoding

`[END_BLOCK, <align32bits>]`

The `END_BLOCK` abbreviation ID specifies the end of the current block record.
Its end is aligned to 32-bits to ensure that the size of the block is an even
multiple of 32-bits.

(Data Records)=

### Data Records

Data records consist of a record code and a number of (up to) 64-bit integer
values.  The interpretation of the code and values is application-specific and
may vary between different block types.  Records can be encoded either using an
unabbrev record, or with an abbreviation.  In the LLVM IR format, for example,
there is a record which encodes the target triple of a module.  The code is
`MODULE_CODE_TRIPLE`, and the values of the record are the ASCII codes for the
characters in the string.

(UNABBREV_RECORD)=

#### UNABBREV_RECORD Encoding

<code>[UNABBREV_RECORD, code<sub>vbr6</sub>, numops<sub>vbr6</sub>, op0<sub>vbr6</sub>, op1<sub>vbr6</sub>, ...]</code>

An `UNABBREV_RECORD` provides a default fallback encoding, which is both
completely general and extremely inefficient.  It can describe an arbitrary
record by emitting the code and operands as VBRs.

For example, emitting an LLVM IR target triple as an unabbreviated record
requires emitting the `UNABBREV_RECORD` abbrevid, a vbr6 for the
`MODULE_CODE_TRIPLE` code, a vbr6 for the length of the string, which is equal
to the number of operands, and a vbr6 for each character.  Because there are no
letters with values less than 32, each letter would need to be emitted as at
least a two-part VBR, which means that each letter would require at least 12
bits.  This is not an efficient encoding, but it is fully general.

(abbreviated record encoding)=

#### Abbreviated Record Encoding

`[<abbrevid>, fields...]`

An abbreviated record is an abbreviation id followed by a set of fields that are
encoded according to the {ref}`abbreviation definition`.  This allows records to be
encoded significantly more densely than records encoded with the
{ref}`UNABBREV_RECORD` type, and allows the abbreviation types to be specified in
the stream itself, which allows the files to be completely self describing.  The
actual encoding of abbreviations is defined below.

The record code, which is the first field of an abbreviated record, may be
encoded in the abbreviation definition (as a literal operand) or supplied in the
abbreviated record (as a Fixed or VBR operand value).

(abbreviation definition)=

### Abbreviations

Abbreviations are an important form of compression for bitstreams.  The idea is
to specify a dense encoding for a class of records once, then use that encoding
to emit many records.  It takes space to emit the encoding into the file, but
the space is recouped (hopefully plus some) when the records that use it are
emitted.

Abbreviations can be determined dynamically per client, per file. Because the
abbreviations are stored in the bitstream itself, different streams of the same
format can contain different sets of abbreviations according to the needs of the
specific stream.  As a concrete example, LLVM IR files usually emit an
abbreviation for binary operators.  If a specific LLVM module contained no or
few binary operators, the abbreviation does not need to be emitted.

(DEFINE_ABBREV)=

#### DEFINE_ABBREV Encoding

<code>[DEFINE_ABBREV, numabbrevops<sub>vbr5</sub>, abbrevop0, abbrevop1, ...]</code>

A `DEFINE_ABBREV` record adds an abbreviation to the list of currently defined
abbreviations in the scope of this block.  This definition only exists inside
this immediate block --- it is not visible in subblocks or enclosing blocks.
Abbreviations are implicitly assigned IDs sequentially starting from 4 (the
first application-defined abbreviation ID).  Any abbreviations defined in a
`BLOCKINFO` record for the particular block type receive IDs first, in order,
followed by any abbreviations defined within the block itself.  Abbreviated data
records reference this ID to indicate what abbreviation they are invoking.

An abbreviation definition consists of the `DEFINE_ABBREV` abbrevid followed
by a VBR that specifies the number of abbrev operands, then the abbrev operands
themselves.  Abbreviation operands come in three forms.  They all start with a
single bit that indicates whether the abbrev operand is a literal operand (when
the bit is 1) or an encoding operand (when the bit is 0).

1. Literal operands --- <code>[1<sub>1</sub>, litvalue<sub>vbr8</sub>]</code> --- Literal operands specify that the value in
   the result is always a single specific value.  This specific value is emitted
   as a vbr8 after the bit indicating that it is a literal operand.

1. Encoding info without data --- <code>[0<sub>1</sub>, encoding<sub>3</sub>]</code> --- Operand encodings that do not have extra data
   are just emitted as their code.

1. Encoding info with data --- <code>[0<sub>1</sub>, encoding<sub>3</sub>, value<sub>vbr5</sub>]</code> --- Operand encodings that do
   have extra data are emitted as their code, followed by the extra data.

The possible operand encodings are:

* Fixed (code 1): The field should be emitted as a {ref}`fixed-width value`, whose
  width is specified by the operand's extra data.

* VBR (code 2): The field should be emitted as a {ref}`variable-width value`, whose
  width is specified by the operand's extra data.

* Array (code 3): This field is an array of values.  The array operand has no
  extra data, but expects another operand to follow it, indicating the element
  type of the array.  When reading an array in an abbreviated record, the first
  integer is a vbr6 that indicates the array length, followed by the encoded
  elements of the array.  An array may only occur as the last operand of an
  abbreviation (except for the one final operand that gives the array's
  type).

* Char6 (code 4): This field should be emitted as a {ref}`char6-encoded value`.
  This operand type takes no extra data. Char6 encoding is normally used as an
  array element type.

* Blob (code 5): This field is emitted as a vbr6, followed by padding to a
  32-bit boundary (for alignment) and an array of 8-bit objects.  The array of
  bytes is further followed by tail padding to ensure that its total length is a
  multiple of 4 bytes.  This makes it very efficient for the reader to decode
  the data without having to make a copy of it: it can use a pointer to the data
  in the mapped in file and poke directly at it.  A blob may only occur as the
  last operand of an abbreviation.

For example, target triples in LLVM modules are encoded as a record of the form
`[TRIPLE, 'a', 'b', 'c', 'd']`.  Consider if the bitstream emitted the
following abbrev entry:

```
[0, Fixed, 4]
[0, Array]
[0, Char6]

```
When emitting a record with this abbreviation, the above entry would be emitted
as:

<code>[4<sub>abbrevwidth</sub>, 2<sub>4</sub>, 4<sub>vbr6</sub>, 0<sub>6</sub>, 1<sub>6</sub>, 2<sub>6</sub>, 3<sub>6</sub>]</code>

These values are:

1. The first value, 4, is the abbreviation ID for this abbreviation.

1. The second value, 2, is the record code for `TRIPLE` records within LLVM IR
   file `MODULE_BLOCK` blocks.

1. The third value, 4, is the length of the array.

1. The rest of the values are the char6 encoded values for `"abcd"`.

With this abbreviation, the triple is emitted with only 37 bits (assuming a
abbrev id width of 3).  Without the abbreviation, significantly more space would
be required to emit the target triple.  Also, because the `TRIPLE` value is
not emitted as a literal in the abbreviation, the abbreviation can also be used
for any other string value.

(standard blocks)=
(standard block)=

### Standard Blocks

In addition to the basic block structure and record encodings, the bitstream
also defines specific built-in block types.  These block types specify how the
stream is to be decoded or other metadata.  In the future, new standard blocks
may be added.  Block IDs 0-7 are reserved for standard blocks.

(BLOCKINFO)=

#### #0 - BLOCKINFO Block

The `BLOCKINFO` block allows the description of metadata for other blocks.
The currently specified records are:

```
[SETBID (#1), blockid]
[DEFINE_ABBREV, ...]
[BLOCKNAME, ...name...]
[SETRECORDNAME, RecordID, ...name...]

```
The `SETBID` record (code 1) indicates which block ID is being described.
`SETBID` records can occur multiple times throughout the block to change which
block ID is being described.  There must be a `SETBID` record prior to any
other records.

Standard `DEFINE_ABBREV` records can occur inside `BLOCKINFO` blocks, but
unlike their occurrence in normal blocks, the abbreviation is defined for blocks
matching the block ID we are describing, *not* the `BLOCKINFO` block
itself.  The abbreviations defined in `BLOCKINFO` blocks receive abbreviation
IDs as described in {ref}`DEFINE_ABBREV`.

The `BLOCKNAME` record (code 2) can optionally occur in this block.  The
elements of the record are the bytes of the string name of the block.
llvm-bcanalyzer can use this to dump out bitcode files symbolically.

The `SETRECORDNAME` record (code 3) can also optionally occur in this block.
The first operand value is a record ID number, and the rest of the elements of
the record are the bytes for the string name of the record.  llvm-bcanalyzer can
use this to dump out bitcode files symbolically.

Note that although the data in `BLOCKINFO` blocks is described as "metadata,"
the abbreviations they contain are essential for parsing records from the
corresponding blocks.  It is not safe to skip them.

(wrapper)=

## Bitcode Wrapper Format

Bitcode files for LLVM IR may optionally be wrapped in a simple wrapper
structure.  This structure contains a simple header that indicates the offset
and size of the embedded BC file.  This allows additional information to be
stored alongside the BC file.  The structure of this file header is:

<code>[Magic<sub>32</sub>, Version<sub>32</sub>, Offset<sub>32</sub>, Size<sub>32</sub>, CPUType<sub>32</sub>]</code>

Each of the fields are 32-bit fields stored in little endian form (as with the
rest of the bitcode file fields).  The Magic number is always `0x0B17C0DE` and
the version is currently always `0`.  The Offset field is the offset in bytes
to the start of the bitcode stream in the file, and the Size field is the size
in bytes of the stream. CPUType is a target-specific value that can be used to
encode the CPU of the target.

(native object file)=

## Native Object File Wrapper Format

Bitcode files for LLVM IR may also be wrapped in a native object file
(i.e., ELF, COFF, Mach-O).  The bitcode must be stored in a section of the object
file named `__LLVM,__bitcode` for Mach-O or `.llvmbc` for the other object
formats. ELF objects additionally support a `.llvm.lto` section for
{doc}`FatLTO`, which contains bitcode suitable for LTO compilation (i.e., bitcode
that has gone through a pre-link LTO pipeline).  The `.llvmbc` section
predates FatLTO support in LLVM, and may not always contain bitcode that is
suitable for LTO (i.e., from `-fembed-bitcode`).  The wrapper format is useful
for accommodating LTO in compilation pipelines where intermediate objects must
be native object files which contain metadata in other sections.

Not all tools support this format.  For example, lld and the gold plugin will
ignore the `.llvmbc` section when linking object files, but can use
`.llvm.lto` sections when passed the correct command-line options.

(encoding of LLVM IR)=

## LLVM IR Encoding

LLVM IR is encoded into a bitstream by defining blocks and records.  It uses
blocks for things like constant pools, functions, symbol tables, etc.  It uses
records for things like instructions, global variable descriptors, type
descriptions, etc.  This document does not describe the set of abbreviations
that the writer uses, as these are fully self-described in the file, and the
reader is not allowed to build in any knowledge of this.

### Basics

#### LLVM IR Magic Number

The magic number for LLVM IR files is:

<code>['B'<sub>8</sub>, 'C'<sub>8</sub>, 0x0<sub>4</sub>, 0xC<sub>4</sub>, 0xE<sub>4</sub>, 0xD<sub>4</sub>]</code>

(Signed VBRs)=

#### Signed VBRs

{ref}`variable-width value` encoding is an efficient way to encode arbitrary sized
unsigned values, but is an extremely inefficient for encoding signed values, as
signed values are otherwise treated as maximally large unsigned values.

As such, signed VBR values of a specific width are emitted as follows:

* Positive values are emitted as VBRs of the specified width, but with their
  value shifted left by one.

* Negative values are emitted as VBRs of the specified width, but the negated
  value is shifted left by one, and the low bit is set.

With this encoding, small positive and small negative values can both be emitted
efficiently. Signed VBR encoding is used in `CST_CODE_INTEGER` and
`CST_CODE_WIDE_INTEGER` records within `CONSTANTS_BLOCK` blocks.
It is also used for phi instruction operands in {ref}`MODULE_CODE_VERSION` 1.

#### LLVM IR Blocks

LLVM IR is defined with the following blocks:

* 8 --- {ref}`MODULE_BLOCK` --- This is the top-level block that contains the entire
  module, and describes a variety of per-module information.

* 9 --- {ref}`PARAMATTR_BLOCK` --- This enumerates the parameter attributes.

* 10 --- {ref}`PARAMATTR_GROUP_BLOCK` --- This describes the attribute group table.

* 11 --- {ref}`CONSTANTS_BLOCK` --- This describes constants for a module or
  function.

* 12 --- {ref}`FUNCTION_BLOCK` --- This describes a function body.

* 14 --- {ref}`VALUE_SYMTAB_BLOCK` --- This describes a value symbol table.

* 15 --- {ref}`METADATA_BLOCK` --- This describes metadata items.

* 16 --- {ref}`METADATA_ATTACHMENT` --- This contains records associating metadata
  with function instruction values.

* 17 --- {ref}`TYPE_BLOCK` --- This describes all of the types in the module.

* 23 --- {ref}`STRTAB_BLOCK` --- The bitcode file's string table.

(MODULE_BLOCK)=

### MODULE_BLOCK Contents

The `MODULE_BLOCK` block (id 8) is the top-level block for LLVM bitcode files,
and each module in a bitcode file must contain exactly one. A bitcode file with
multi-module bitcode is valid. In addition to records (described below)
containing information about the module, a `MODULE_BLOCK` block may contain
the following sub-blocks:

* {ref}`BLOCKINFO`
* {ref}`PARAMATTR_BLOCK`
* {ref}`PARAMATTR_GROUP_BLOCK`
* {ref}`TYPE_BLOCK`
* {ref}`VALUE_SYMTAB_BLOCK`
* {ref}`CONSTANTS_BLOCK`
* {ref}`FUNCTION_BLOCK`
* {ref}`METADATA_BLOCK`

(MODULE_CODE_VERSION)=

#### MODULE_CODE_VERSION Record

`[VERSION, version#]`

The `VERSION` record (code 1) contains a single value indicating the format
version. Versions 0, 1, and 2 are supported at this time. The difference between
version 0 and 1 is in the encoding of instruction operands in
each {ref}`FUNCTION_BLOCK`.

In version 0, each value defined by an instruction is assigned an ID
unique to the function. Function-level value IDs are assigned starting from
`NumModuleValues` since they share the same namespace as module-level
values. The value enumerator resets after each function. When a value is
an operand of an instruction, the value ID is used to represent the operand.
For large functions or large modules, these operand values can be large.

The encoding in version 1 attempts to avoid large operand values
in common cases. Instead of using the value ID directly, operands are
encoded as relative to the current instruction. Thus, if an operand
is the value defined by the previous instruction, the operand
will be encoded as 1.

For example, instead of

```
##n = load #n-1
##n+1 = icmp eq #n, #const0
br #n+1, label #(bb1), label #(bb2)

```
version 1 will encode the instructions as

```
##n = load #1
##n+1 = icmp eq #1, (#n+1)-#const0
br #1, label #(bb1), label #(bb2)

```
Note in the example that operands which are constants also use
the relative encoding, while operands like basic block labels
do not use the relative encoding.

Forward references will result in a negative value.
This can be inefficient, as operands are normally encoded
as unsigned VBRs. However, forward references are rare, except in the
case of phi instructions. For phi instructions, operands are encoded as
{ref}`Signed VBRs` to deal with forward references.

In version 2, the meaning of module records `FUNCTION`, `GLOBALVAR`,
`ALIAS`, `IFUNC` and `COMDAT` change such that the first two operands
specify an offset and size of a string in a string table (see `STRTAB_BLOCK
Contents`_), the function name is removed from the `FNENTRY` record in the
value symbol table, and the top-level `VALUE_SYMTAB_BLOCK` may only contain
`FNENTRY` records.

#### MODULE_CODE_TRIPLE Record

`[TRIPLE, ...string...]`

The `TRIPLE` record (code 2) contains a variable number of values representing
the bytes of the `target triple` specification string.

#### MODULE_CODE_DATALAYOUT Record

`[DATALAYOUT, ...string...]`

The `DATALAYOUT` record (code 3) contains a variable number of values
representing the bytes of the `target datalayout` specification string.

#### MODULE_CODE_ASM Record

`[ASM, ...string...]`

The `ASM` record (code 4) contains a variable number of values representing
the bytes of `module asm` strings, with individual assembly blocks separated
by newline (ASCII 10) characters.

(MODULE_CODE_SECTIONNAME)=

#### MODULE_CODE_SECTIONNAME Record

`[SECTIONNAME, ...string...]`

The `SECTIONNAME` record (code 5) contains a variable number of values
representing the bytes of a single section name string. There should be one
`SECTIONNAME` record for each section name referenced (e.g., in global
variable or function `section` attributes) within the module. These records
can be referenced by the 1-based index in the *section* fields of `GLOBALVAR`
or `FUNCTION` records.

#### MODULE_CODE_DEPLIB Record

`[DEPLIB, ...string...]`

The `DEPLIB` record (code 6) contains a variable number of values representing
the bytes of a single dependent library name string, one of the libraries
mentioned in a `deplibs` declaration.  There should be one `DEPLIB` record
for each library name referenced.

#### MODULE_CODE_GLOBALVAR Record

`[GLOBALVAR, strtab offset, strtab size, pointer type, isconst, initid, linkage, alignment, section, visibility, threadlocal, unnamed_addr, externally_initialized, dllstorageclass, comdat, attributes, preemptionspecifier]`

The `GLOBALVAR` record (code 7) marks the declaration or definition of a
global variable. The operand fields are:

* *strtab offset*, *strtab size*: Specifies the name of the global variable.
  See {ref}`STRTAB_BLOCK`.

* *pointer type*: The type index of the pointer type used to point to this
  global variable

* *isconst*: Non-zero if the variable is treated as constant within the module,
  or zero if it is not

* *initid*: If non-zero, the value index of the initializer for this variable,
  plus 1.

(linkage type)=

* *linkage*: An encoding of the linkage type for this variable:

  * `external`: code 0
  * `weak`: code 1
  * `appending`: code 2
  * `internal`: code 3
  * `linkonce`: code 4
  * `dllimport`: code 5
  * `dllexport`: code 6
  * `extern_weak`: code 7
  * `common`: code 8
  * `private`: code 9
  * `weak_odr`: code 10
  * `linkonce_odr`: code 11
  * `available_externally`: code 12
  * deprecated : code 13
  * deprecated : code 14

* alignment*: The logarithm base 2 of the variable's requested alignment, plus 1

* *section*: If non-zero, the 1-based section index in the table of
  {ref}`MODULE_CODE_SECTIONNAME` entries.

(visibility)=

* *visibility*: If present, an encoding of the visibility of this variable:

  * `default`: code 0
  * `hidden`: code 1
  * `protected`: code 2

(bcthreadlocal)=

* *threadlocal*: If present, an encoding of the thread local storage mode of the
  variable:

  * `not thread local`: code 0
  * `thread local; default TLS model`: code 1
  * `localdynamic`: code 2
  * `initialexec`: code 3
  * `localexec`: code 4

(bcunnamedaddr)=

* *unnamed_addr*: If present, an encoding of the `unnamed_addr` attribute of this
  variable:

  * not `unnamed_addr`: code 0
  * `unnamed_addr`: code 1
  * `local_unnamed_addr`: code 2

(bcdllstorageclass)=

* *dllstorageclass*: If present, an encoding of the DLL storage class of this variable:

  * `default`: code 0
  * `dllimport`: code 1
  * `dllexport`: code 2

* *comdat*: An encoding of the COMDAT of this function

* *attributes*: If nonzero, the 1-based index into the table of AttributeLists.

(bcpreemptionspecifier)=

* *preemptionspecifier*: If present, an encoding of the runtime preemption specifier of this variable:

  * `dso_preemptable`: code 0
  * `dso_local`: code 1

(FUNCTION)=

#### MODULE_CODE_FUNCTION Record

`[FUNCTION, strtab offset, strtab size, type, callingconv, isproto, linkage, paramattr, alignment, section, visibility, gc, prologuedata, dllstorageclass, comdat, prefixdata, personalityfn, preemptionspecifier]`

The `FUNCTION` record (code 8) marks the declaration or definition of a
function. The operand fields are:

* *strtab offset*, *strtab size*: Specifies the name of the function.
  See {ref}`STRTAB_BLOCK`.

* *type*: The type index of the function type describing this function

* *callingconv*: The calling convention number:
  * `ccc`: code 0
  * `fastcc`: code 8
  * `coldcc`: code 9
  * `anyregcc`: code 13
  * `preserve_mostcc`: code 14
  * `preserve_allcc`: code 15
  * `swiftcc` : code 16
  * `cxx_fast_tlscc`: code 17
  * `tailcc` : code 18
  * `cfguard_checkcc` : code 19
  * `swifttailcc` : code 20
  * `x86_stdcallcc`: code 64
  * `x86_fastcallcc`: code 65
  * `arm_apcscc`: code 66
  * `arm_aapcscc`: code 67
  * `arm_aapcs_vfpcc`: code 68

* isproto*: Non-zero if this entry represents a declaration rather than a
  definition

* *linkage*: An encoding of the {ref}`linkage type <linkage type>` for this function

* *paramattr*: If nonzero, the 1-based parameter attribute index into the table
  of {ref}`PARAMATTR_CODE_ENTRY` entries.

* *alignment*: The logarithm base 2 of the function's requested alignment, plus
  1

* *section*: If non-zero, the 1-based section index in the table of
  {ref}`MODULE_CODE_SECTIONNAME` entries.

* *visibility*: An encoding of the {ref}`visibility <visibility>` of this function

* *gc*: If present and nonzero, the 1-based garbage collector index in the table
  of {ref}`MODULE_CODE_GCNAME` entries.

* *unnamed_addr*: If present, an encoding of the
  {ref}`unnamed_addr<bcunnamedaddr>` attribute of this function

* *prologuedata*: If non-zero, the value index of the prologue data for this function,
  plus 1.

* *dllstorageclass*: An encoding of the
  {ref}`dllstorageclass<bcdllstorageclass>` of this function

* *comdat*: An encoding of the COMDAT of this function

* *prefixdata*: If non-zero, the value index of the prefix data for this function,
  plus 1.

* *personalityfn*: If non-zero, the value index of the personality function for this function,
  plus 1.

* *preemptionspecifier*: If present, an encoding of the {ref}`runtime preemption specifier<bcpreemptionspecifier>`  of this function.

#### MODULE_CODE_ALIAS Record

`[ALIAS, strtab offset, strtab size, alias type, aliasee val#, linkage, visibility, dllstorageclass, threadlocal, unnamed_addr, preemptionspecifier]`

The `ALIAS` record (code 9) marks the definition of an alias. The operand
fields are

* *strtab offset*, *strtab size*: Specifies the name of the alias.
  See {ref}`STRTAB_BLOCK`.

* *alias type*: The type index of the alias

* *aliasee val#*: The value index of the aliased value

* *linkage*: An encoding of the {ref}`linkage type <linkage type>` for this alias

* *visibility*: If present, an encoding of the {ref}`visibility <visibility>` of the alias

* *dllstorageclass*: If present, an encoding of the
  {ref}`dllstorageclass<bcdllstorageclass>` of the alias

* *threadlocal*: If present, an encoding of the
  {ref}`thread local property<bcthreadlocal>` of the alias

* *unnamed_addr*: If present, an encoding of the
  {ref}`unnamed_addr<bcunnamedaddr>` attribute of this alias

* *preemptionspecifier*: If present, an encoding of the {ref}`runtime preemption specifier<bcpreemptionspecifier>`  of this alias.

(MODULE_CODE_GCNAME)=

#### MODULE_CODE_GCNAME Record

`[GCNAME, ...string...]`

The `GCNAME` record (code 11) contains a variable number of values
representing the bytes of a single garbage collector name string. There should
be one `GCNAME` record for each garbage collector name referenced in function
`gc` attributes within the module. These records can be referenced by 1-based
index in the *gc* fields of `FUNCTION` records.

(PARAMATTR_BLOCK)=

### PARAMATTR_BLOCK Contents

The `PARAMATTR_BLOCK` block (id 9) contains a table of entries describing the
attributes of function parameters. These entries are referenced by 1-based index
in the *paramattr* field of module block {ref}`FUNCTION` records, or within the
*attr* field of function block `INST_INVOKE` and `INST_CALL` records.

Entries within `PARAMATTR_BLOCK` are constructed to ensure that each is unique
(i.e., no two indices represent equivalent attribute lists).

(PARAMATTR_CODE_ENTRY)=

#### PARAMATTR_CODE_ENTRY Record

`[ENTRY, attrgrp0, attrgrp1, ...]`

The `ENTRY` record (code 2) contains a variable number of values describing a
unique set of function parameter attributes. Each *attrgrp* value is used as a
key with which to look up an entry in the attribute group table described
in the `PARAMATTR_GROUP_BLOCK` block.

(PARAMATTR_CODE_ENTRY_OLD)=

#### PARAMATTR_CODE_ENTRY_OLD Record

```{note}
This is a legacy encoding for attributes, produced by LLVM versions 3.2 and
earlier. It is guaranteed to be understood by the current LLVM version, as
specified in the {ref}`IR backwards compatibility` policy.

```
`[ENTRY, paramidx0, attr0, paramidx1, attr1...]`

The `ENTRY` record (code 1) contains an even number of values describing a
unique set of function parameter attributes. Each *paramidx* value indicates
which set of attributes is represented, with 0 representing the return value
attributes, 0xFFFFFFFF representing function attributes, and other values
representing 1-based function parameters. Each *attr* value is a bitmap with the
following interpretation:

* bit 0: `zeroext`
* bit 1: `signext`
* bit 2: `noreturn`
* bit 3: `inreg`
* bit 4: `sret`
* bit 5: `nounwind`
* bit 6: `noalias`
* bit 7: `byval`
* bit 8: `nest`
* bit 9: `readnone`
* bit 10: `readonly`
* bit 11: `noinline`
* bit 12: `alwaysinline`
* bit 13: `optsize`
* bit 14: `ssp`
* bit 15: `sspreq`
* bits 16-31: `align n`
* bit 32: `nocapture`
* bit 33: `noredzone`
* bit 34: `noimplicitfloat`
* bit 35: `naked`
* bit 36: `inlinehint`
* bits 37-39: `alignstack n`, represented as the logarithm
  base 2 of the requested alignment, plus 1

(PARAMATTR_GROUP_BLOCK)=

### PARAMATTR_GROUP_BLOCK Contents

The `PARAMATTR_GROUP_BLOCK` block (id 10) contains a table of entries
describing the attribute groups present in the module. These entries can be
referenced within `PARAMATTR_CODE_ENTRY` entries.

(PARAMATTR_GRP_CODE_ENTRY)=

#### PARAMATTR_GRP_CODE_ENTRY Record

`[ENTRY, grpid, paramidx, attr0, attr1, ...]`

The `ENTRY` record (code 3) contains *grpid* and *paramidx* values, followed
by a variable number of values describing a unique group of attributes. The
*grpid* value is a unique key for the attribute group, which can be referenced
within `PARAMATTR_CODE_ENTRY` entries. The *paramidx* value indicates which
set of attributes is represented, with 0 representing the return value
attributes, 0xFFFFFFFF representing function attributes, and other values
representing 1-based function parameters.

Each *attr* is itself represented as a variable number of values:

`kind, key [, ...], [value [, ...]]`

Each attribute is either a well-known LLVM attribute (possibly with an integer
value associated with it), or an arbitrary string (possibly with an arbitrary
string value associated with it). The *kind* value is an integer code
distinguishing between these possibilities:

* code 0: well-known attribute
* code 1: well-known attribute with an integer value
* code 3: string attribute
* code 4: string attribute with a string value

For well-known attributes (code 0 or 1), the *key* value is an integer code
identifying the attribute. For attributes with an integer argument (code 1),
the *value* value indicates the argument.

For string attributes (code 3 or 4), the *key* value is actually a variable
number of values representing the bytes of a null-terminated string. For
attributes with a string argument (code 4), the *value* value is similarly a
variable number of values representing the bytes of a null-terminated string.

The integer codes are mapped to attributes as described in the
`AttributeKindCodes` enumeration in the file [LLVMBitCodes.h](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Bitcode/LLVMBitCodes.h).

For example:

```
enum AttributeKindCodes {
  // = 0 is unused
  ATTR_KIND_ALIGNMENT = 1,
  ATTR_KIND_ALWAYS_INLINE = 2,
  ...
  }

```
Correspond to:

* code 1: `align(<n>)`
* code 2: `alwaysinline`

The mappings between the enumeration and the attribute name string may be found
in the file [Attributes.td](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/Attributes.td).

```{note}
The `allocsize` attribute has a special encoding for its arguments. Its two
arguments, which are 32-bit integers, are packed into one 64-bit integer value
(i.e., `(EltSizeParam << 32) | NumEltsParam`), with `NumEltsParam` taking on
the sentinel value -1 if it is not specified.

```
```{note}
The `vscale_range` attribute has a special encoding for its arguments. Its two
arguments, which are 32-bit integers, are packed into one 64-bit integer value
(i.e., `(Min << 32) | Max`), with `Max` taking on the value of `Min` if
it is not specified.

```
(TYPE_BLOCK)=

### TYPE_BLOCK Contents

The `TYPE_BLOCK` block (id 17) contains records which constitute a table of
type operator entries used to represent types referenced within an LLVM
module. Each record (with the exception of {ref}`NUMENTRY`) generates a single type
table entry, which may be referenced by 0-based index from instructions,
constants, metadata, type symbol table entries, or other type operator records.

Entries within `TYPE_BLOCK` are constructed to ensure that each entry is
unique (i.e., no two indices represent structurally equivalent types).

(TYPE_CODE_NUMENTRY)=
(NUMENTRY)=

#### TYPE_CODE_NUMENTRY Record

`[NUMENTRY, numentries]`

The `NUMENTRY` record (code 1) contains a single value which indicates the
total number of type code entries in the type table of the module. If present,
`NUMENTRY` should be the first record in the block.

#### TYPE_CODE_VOID Record

`[VOID]`

The `VOID` record (code 2) adds a `void` type to the type table.

#### TYPE_CODE_HALF Record

`[HALF]`

The `HALF` record (code 10) adds a `half` (16-bit floating point) type to
the type table.

#### TYPE_CODE_BFLOAT Record

`[BFLOAT]`

The `BFLOAT` record (code 23) adds a `bfloat` (16-bit brain floating point)
type to the type table.

#### TYPE_CODE_FLOAT Record

`[FLOAT]`

The `FLOAT` record (code 3) adds a `float` (32-bit floating point) type to
the type table.

#### TYPE_CODE_DOUBLE Record

`[DOUBLE]`

The `DOUBLE` record (code 4) adds a `double` (64-bit floating point) type to
the type table.

#### TYPE_CODE_LABEL Record

`[LABEL]`

The `LABEL` record (code 5) adds a `label` type to the type table.

#### TYPE_CODE_OPAQUE Record

`[OPAQUE]`

The `OPAQUE` record (code 6) adds an `opaque` type to the type table, with
a name defined by a previously encountered `STRUCT_NAME` record. Note that
distinct `opaque` types are not unified.

#### TYPE_CODE_INTEGER Record

`[INTEGER, width]`

The `INTEGER` record (code 7) adds an integer type to the type table. The
single *width* field indicates the width of the integer type.

#### TYPE_CODE_POINTER Record

`[POINTER, pointee type, address space]`

The `POINTER` record (code 8) adds a pointer type to the type table. The
operand fields are:

* *pointee type*: The type index of the pointed-to type

* *address space*: If supplied, the target-specific numbered address space where
  the pointed-to object resides. Otherwise, the default address space is zero.

#### TYPE_CODE_FUNCTION_OLD Record

```{note}
This is a legacy encoding for functions, produced by LLVM versions 3.0 and
earlier. It is guaranteed to be understood by the current LLVM version, as
specified in the {ref}`IR backwards compatibility` policy.

```
`[FUNCTION_OLD, vararg, ignored, retty, ...paramty... ]`

The `FUNCTION_OLD` record (code 9) adds a function type to the type table.
The operand fields are:

* *vararg*: Non-zero if the type represents a varargs function

* *ignored*: This value field is present for backward compatibility only, and is
  ignored

* *retty*: The type index of the function's return type

* *paramty*: Zero or more type indices representing the parameter types of the
  function

#### TYPE_CODE_ARRAY Record

`[ARRAY, numelts, eltty]`

The `ARRAY` record (code 11) adds an array type to the type table.  The
operand fields are:

* *numelts*: The number of elements in arrays of this type

* *eltty*: The type index of the array element type

#### TYPE_CODE_VECTOR Record

`[VECTOR, numelts, eltty]`

The `VECTOR` record (code 12) adds a vector type to the type table.  The
operand fields are:

* *numelts*: The number of elements in vectors of this type

* *eltty*: The type index of the vector element type

#### TYPE_CODE_X86_FP80 Record

`[X86_FP80]`

The `X86_FP80` record (code 13) adds an `x86_fp80` (80-bit floating point)
type to the type table.

#### TYPE_CODE_FP128 Record

`[FP128]`

The `FP128` record (code 14) adds an `fp128` (128-bit floating point) type
to the type table.

#### TYPE_CODE_PPC_FP128 Record

`[PPC_FP128]`

The `PPC_FP128` record (code 15) adds a `ppc_fp128` (128-bit floating point)
type to the type table.

#### TYPE_CODE_METADATA Record

`[METADATA]`

The `METADATA` record (code 16) adds a `metadata` type to the type table.

#### TYPE_CODE_X86_MMX Record

`[X86_MMX]`

The `X86_MMX` record (code 17) is deprecated, and imported as a `<1 x i64>` vector.

#### TYPE_CODE_STRUCT_ANON Record

`[STRUCT_ANON, ispacked, ...eltty...]`

The `STRUCT_ANON` record (code 18) adds a literal struct type to the type
table. The operand fields are:

* *ispacked*: Non-zero if the type represents a packed structure

* *eltty*: Zero or more type indices representing the element types of the
  structure

#### TYPE_CODE_STRUCT_NAME Record

`[STRUCT_NAME, ...string...]`

The `STRUCT_NAME` record (code 19) contains a variable number of values
representing the bytes of a struct name. The next `OPAQUE` or
`STRUCT_NAMED` record will use this name.

#### TYPE_CODE_STRUCT_NAMED Record

`[STRUCT_NAMED, ispacked, ...eltty...]`

The `STRUCT_NAMED` record (code 20) adds an identified struct type to the
type table, with a name defined by a previously encountered `STRUCT_NAME`
record. The operand fields are:

* *ispacked*: Non-zero if the type represents a packed structure

* *eltty*: Zero or more type indices representing the element types of the
  structure

#### TYPE_CODE_FUNCTION Record

`[FUNCTION, vararg, retty, ...paramty... ]`

The `FUNCTION` record (code 21) adds a function type to the type table. The
operand fields are:

* *vararg*: Non-zero if the type represents a varargs function

* *retty*: The type index of the function's return type

* *paramty*: Zero or more type indices representing the parameter types of the
  function

#### TYPE_CODE_X86_AMX Record

`[X86_AMX]`

The `X86_AMX` record (code 24) adds an `x86_amx` type to the type table.

#### TYPE_CODE_TARGET_TYPE Record

`[TARGET_TYPE, num_tys, ...ty_params..., ...int_params... ]`

The `TARGET_TYPE` record (code 26) adds a target extension type to the type
table, with a name defined by a previously encountered `STRUCT_NAME` record.
The operand fields are:

* *num_tys*: The number of parameters that are types (as opposed to integers)

* *ty_params*: Type indices that represent type parameters

* *int_params*: Numbers that correspond to the integer parameters.

(CONSTANTS_BLOCK)=

### CONSTANTS_BLOCK Contents

The `CONSTANTS_BLOCK` block (id 11) ...

(FUNCTION_BLOCK)=

### FUNCTION_BLOCK Contents

The `FUNCTION_BLOCK` block (id 12) ...

In addition to the record types described below, a `FUNCTION_BLOCK` block may
contain the following sub-blocks:

* {ref}`CONSTANTS_BLOCK`
* {ref}`VALUE_SYMTAB_BLOCK`
* {ref}`METADATA_ATTACHMENT`

(VALUE_SYMTAB_BLOCK)=

### VALUE_SYMTAB_BLOCK Contents

The `VALUE_SYMTAB_BLOCK` block (id 14) ...

(METADATA_BLOCK)=

### METADATA_BLOCK Contents

The `METADATA_BLOCK` block (id 15) ...

(METADATA_ATTACHMENT)=

### METADATA_ATTACHMENT Contents

The `METADATA_ATTACHMENT` block (id 16) ...

(STRTAB_BLOCK)=

### STRTAB_BLOCK Contents

The `STRTAB` block (id 23) contains a single record (`STRTAB_BLOB`, id 1)
with a single blob operand containing the bitcode file's string table.

Strings in the string table are not null terminated. A record's *strtab
offset* and *strtab size* operands specify the byte offset and size of a
string within the string table.

The string table is used by all preceding blocks in the bitcode file that are
not succeeded by another intervening `STRTAB` block. Normally a bitcode
file will have a single string table, but it may have more than one if it
was created by binary concatenation of multiple bitcode files.
