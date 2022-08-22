# MLIR Bytecode Format

This documents describes the MLIR bytecode format and its encoding.

[TOC]

## Magic Number

MLIR uses the following four-byte magic number to indicate bytecode files:

'\[‘M’<sub>8</sub>, ‘L’<sub>8</sub>, ‘ï’<sub>8</sub>, ‘R’<sub>8</sub>\]'

In hex:

'\[‘4D’<sub>8</sub>, ‘4C’<sub>8</sub>, ‘EF’<sub>8</sub>, ‘52’<sub>8</sub>\]'

## Format Overview

An MLIR Bytecode file is comprised of a byte stream, with a few simple
structural concepts layered on top.

### Primitives

#### Fixed-Width Integers

```
  byte                  ::= `0x00`...`0xFF`
```

Fixed width integers are unsigned integers of a known byte size. The values are
stored in little-endian byte order.

TODO: Add larger fixed width integers as necessary.

#### Variable-Width Integers

Variable width integers, or `VarInt`s, provide a compact representation for
integers. Each encoded VarInt consists of one to nine bytes, which together
represent a single 64-bit value. The MLIR bytecode utilizes the "PrefixVarInt"
encoding for VarInts. This encoding is a variant of the
[LEB128 ("Little-Endian Base 128")](https://en.wikipedia.org/wiki/LEB128)
encoding, where each byte of the encoding provides up to 7 bits for the value,
with the remaining bit used to store a tag indicating the number of bytes used
for the encoding. This means that small unsigned integers (less than 2^7) may be
stored in one byte, unsigned integers up to 2^14 may be stored in two bytes,
etc.

The first byte of the encoding includes a length prefix in the low bits. This
prefix is a bit sequence of '0's followed by a terminal '1', or the end of the
byte. The number of '0' bits indicate the number of _additional_ bytes, not
including the prefix byte, used to encode the value. All of the remaining bits
in the first byte, along with all of the bits in the additional bytes, provide
the value of the integer. Below are the various possible encodings of the prefix
byte:

```
xxxxxxx1:  7 value bits, the encoding uses 1 byte
xxxxxx10: 14 value bits, the encoding uses 2 bytes
xxxxx100: 21 value bits, the encoding uses 3 bytes
xxxx1000: 28 value bits, the encoding uses 4 bytes
xxx10000: 35 value bits, the encoding uses 5 bytes
xx100000: 42 value bits, the encoding uses 6 bytes
x1000000: 49 value bits, the encoding uses 7 bytes
10000000: 56 value bits, the encoding uses 8 bytes
00000000: 64 value bits, the encoding uses 9 bytes
```

#### Strings

Strings are blobs of characters with an associated length.

### Sections

```
section {
  id: byte
  length: varint
}
```

Sections are a mechanism for grouping data within the bytecode. The enable
delayed processing, which is useful for out-of-order processing of data,
lazy-loading, and more. Each section contains a Section ID and a length (which
allowing for skipping over the section).

TODO: Sections should also carry an optional alignment. Add this when necessary.

## MLIR Encoding

Given the generic structure of MLIR, the bytecode encoding is actually fairly
simplistic. It effectively maps to the core components of MLIR.

### Top Level Structure

The top-level structure of the bytecode contains the 4-byte "magic number", a
version number, a null-terminated producer string, and a list of sections. Each
section is currently only expected to appear once within a bytecode file.

```
bytecode {
  magic: "MLïR",
  version: varint,
  producer: string,
  sections: section[]
}
```

### String Section

```
strings {
  numStrings: varint,
  reverseStringLengths: varint[],
  stringData: byte[]
}
```

The string section contains a table of strings referenced within the bytecode,
more easily enabling string sharing. This section is encoded first with the
total number of strings, followed by the sizes of each of the individual strings
in reverse order. The remaining encoding contains a single blob containing all
of the strings concatenated together.

### Dialect Section

The dialect section of the bytecode contains all of the dialects referenced
within the encoded IR, and some information about the components of those
dialects that were also referenced.

```
dialect_section {
  numDialects: varint,
  dialectNames: varint[],
  opNames: op_name_group[]
}

op_name_group {
  dialect: varint,
  numOpNames: varint,
  opNames: varint[]
}
```

Dialects are encoded as indexes to the name string within the string section.
Operation names are encoded in groups by dialect, with each group containing the
dialect, the number of operation names, and the array of indexes to each name
within the string section.

### Attribute/Type Sections

Attributes and types are encoded using two [sections](#sections), one section
(`attr_type_section`) containing the actual encoded representation, and another
section (`attr_type_offset_section`) containing the offsets of each encoded
attribute/type into the previous section. This structure allows for attributes
and types to always be lazily loaded on demand.

```
attr_type_section {
  attrs: attribute[],
  types: type[]  
}
attr_type_offset_section {
  numAttrs: varint,
  numTypes: varint,
  offsets: attr_type_offset_group[]
}

attr_type_offset_group {
  dialect: varint,
  numElements: varint,
  offsets: varint[] // (offset << 1) | (hasCustomEncoding) 
}

attribute {
  encoding: ...
}
type {
  encoding: ...
}
```

Each `offset` in the `attr_type_offset_section` above is the size of the
encoding for the attribute or type and a flag indicating if the encoding uses
the textual assembly format, or a custom bytecode encoding. We avoid using the
direct offset into the `attr_type_section`, as a smaller relative offsets
provides more effective compression. Attributes and types are grouped by
dialect, with each `attr_type_offset_group` in the offset section containing the
corresponding parent dialect, number of elements, and offsets for each element
within the group.

#### Attribute/Type Encodings

In the abstract, an attribute/type is encoded in one of two possible ways: via
its assembly format, or via a custom dialect defined encoding.

##### Assembly Format Fallback

In the case where a dialect does not define a method for encoding the attribute
or type, the textual assembly format of that attribute or type is used as a
fallback. For example, a type of `!bytecode.type` would be encoded as the null
terminated string "!bytecode.type". This ensures that every attribute and type
may be encoded, even if the owning dialect has not yet opted in to a more
efficient serialization.

TODO: We shouldn't redundantly encode the dialect name here, we should use a
reference to the parent dialect instead.

##### Dialect Defined Encoding

TODO: This is not yet supported.

### IR Section

The IR section contains the encoded form of operations within the bytecode.

#### Operation Encoding

```
op {
  name: varint,
  encodingMask: byte,
  location: varint,

  attrDict: varint?,

  numResults: varint?,
  resultTypes: varint[],

  numOperands: varint?,
  operands: varint[],

  numSuccessors: varint?,
  successors: varint[],

  regionEncoding: varint?, // (numRegions << 1) | (isIsolatedFromAbove) 
  regions: region[]
}
```

The encoding of an operation is important because this is generally the most
commonly appearing structure in the bytecode. A single encoding is used for
every type of operation. Given this prevelance, many of the fields of an
operation are optional. The `encodingMask` field is a bitmask which indicates
which of the components of the operation are present.

##### Location

The location is encoded as the index of the location within the attribute table.

##### Attributes

If the operation has attribues, the index of the operation attribute dictionary
within the attribute table is encoded.

##### Results

If the operation has results, the number of results and the indexes of the
result types within the type table are encoded.

##### Operands

If the operation has operands, the number of operands and the value index of
each operand is encoded. This value index is the relative ordering of the
definition of that value from the start of the first ancestor isolated region.

##### Successors

If the operation has successors, the number of successors and the indexes of the
successor blocks within the parent region are encoded.

##### Regions

If the operation has regions, the number of regions and if the regions are
isolated from above are encoded together in a single varint. Afterwards, each
region is encoded inline.

#### Region Encoding

```
region {
  numBlocks: varint,

  numValues: varint?,
  blocks: block[]
}
```

A region is encoded first with the number of blocks within. If the region is
non-empty, the number of values defined directly within the region are encoded,
followed by the blocks of the region.

#### Block Encoding

```
block {
  encoding: varint, // (numOps << 1) | (hasBlockArgs)
  arguments: block_arguments?, // Optional based on encoding
  ops : op[]
}

block_arguments {
  numArgs: varint?,
  args: block_argument[]
}

block_argument {
  typeIndex: varint,
  location: varint
}
```

A block is encoded with an array of operations and block arguments. The first
field is an encoding that combines the number of operations in the block, with a
flag indicating if the block has arguments.
