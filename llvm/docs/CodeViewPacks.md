# CodeView packs (experimental format version 1)

This document specifies an experimental feature: CodeView pack records.
The records use ordinary C13 streams and COFF objects. A packed object's
`.debug$T` has its own signature, and its `.drectve` names a sentinel symbol,
so that tools that do not know packs reject it (see Interoperability). These
values are local to this fork and are not an upstream allocation.

CodeView already shares types across objects in two ways: `LF_PRECOMP`, which
refers to the types of a precompiled header object, and `LF_TYPESERVER2`,
which refers to a PDB that `/Zi` builds fill through mspdbsrv. CodeView packs
share records from any headers and deduplicate them after compilation. Each
packed object names the content ID of the one pack it refers to and can be
restored byte for byte, so it can be stored in a build cache or transferred
between workers.

## Record values and versioning

| Record | Value | Stream |
|---|---:|---|
| `LF_CVPACK_LIST` | `0x1700` | `.debug$T` |
| `LF_CVPACK_ORIGIN` | `0x1701` | `.debug$T` |
| `LF_CVPACK_MAP` | `0x1702` | `.debug$T` |
| `S_CVPACK_MAP` | `0x1200` | `.debug$S` symbols subsection |
| files subsection | `0x80000f10` | pack `.debug$S` |
| functions subsection | `0x80000f11` | pack `.debug$S` |
| function reference subsection | `0x80000f12` | object COMDAT `.debug$S` |
| packed type stream signature | `0x31505643` (`"CVP1"`) | object `.debug$T` |

The three subsection kinds carry the CodeView ignore bit (`0x80000000`), so
readers that do not know them skip them. They have no version header.

All integer fields are little-endian. A type record or symbol record has its
normal CodeView record prefix. Its payload begins with this 8-byte header:

```text
u16 version = 1
u16 flags = 0
u32 body_size
u8  body[body_size]
```

`body_size` excludes the CodeView record prefix and any CodeView alignment
padding. Version 1 requires zero flags and rejects unknown versions. A change
to the meaning or layout of a body requires a new version; version 1 has no
implicit extension fields. BLAKE3-256 is the fixed content-id algorithm in
version 1. A pack's content id is BLAKE3-256 of the entire COFF file.

The `.debug$T` of a packed object starts with the signature `0x31505643`
instead of the C13 signature `4`; packs keep `4`.

`LF_CVPACK_LIST` is required as the first type record in every packed object.
`LF_CVPACK_ORIGIN`, when present, is the next type record and appears only in
a rewritten object. One or more `LF_CVPACK_MAP` records follow these headers.
All of these leading records consume no type index. Ordinary records after
them retain the logical indices defined below. A pack holds none of these
records: packs do not refer to other packs.

## Pack list

The `LF_CVPACK_LIST` body is:

```text
u32 pack_count
byte pack_ids[pack_count][32]
```

In version 1, `pack_count` is 1: an object refers to one pack. The count, and
the slot fields of range maps and function references, which must be 0, are
kept so that a later version can let an object refer to several packs.

## Range maps

The bodies of all consecutive `LF_CVPACK_MAP` records in `.debug$T` are
concatenated in stream order. A chunk is at most `0xfef4` body bytes and ends
between entries. This keeps the complete record, including its prefix and
version header, within CodeView's `0xff00` record limit. The same rule applies
to consecutive `S_CVPACK_MAP` records at the start of a `.debug$S` symbols
subsection. Each map body is a sequence of entries, with no count or
terminator:

```text
ULEB128 skip
ULEB128 slot
ULEB128 count
zigzag-ULEB128 first_delta
```

ULEB values use the shortest encoding. `skip` is the number of inline records
or symbols since the previous mapped range. The first `skip` is measured from
the beginning of the logical stream. After applying `skip`, the range occupies
the next `count` logical positions. `count` must be nonzero. Inline records
after the last entry are implicit.

`slot` must be 0. The map tracks the end of its previous range, initially
`0x1000` for types and `0` for symbols. The range's first pack index is
`previous_end + first_delta`; its end becomes `first + count`. The range must
fit in the pack. A map entry's `first_delta` is signed 64-bit zigzag ULEB128;
the resolved index and end must fit the index space for the stream.

Leading CodeView pack records do not consume indices. The logical type stream
starts at `0x1000`; each mapped range consumes `count` indices and each inline
type record consumes one. The logical symbol stream is scoped to its
subsection; mapped ranges consume `count` symbols and inline symbols consume
one. A symbol map refers to the pack named by the `LF_CVPACK_LIST` of the
same object's `.debug$T` stream.

Map chunks occur only in the leading type-record sequence or at the beginning
of the symbol subsection. Chunks for one map must be contiguous and each chunk
must end at an entry boundary. A reader rejects malformed ULEB values,
overflow, invalid slots, zero counts, and maps that do not fit the logical
stream.

## CodeView pack COFF file

A pack is a COFF object with no COFF symbol table and two sections,
`.debug$T` and `.debug$S`, plus an optional `.debug$H`. The first two use
ordinary C13 streams with signature `4`. The pack has no relocations, line
numbers, code or data. Its machine is that of the objects packed against it:
the producer packs the objects of one machine at a time. A pack may have a
`.debug$H` section in the format
objects use: the 8-byte header with the BLAKE3 algorithm, then the 8-byte
global hash of each type record in `.debug$T`, in order. Readers use those
hashes instead of computing them; the producer writes them for every base.
`.debug$T` contains only the pack's type records, which reference nothing
outside the pack, numbered from `0x1000`. `.debug$S` contains one symbols
subsection with `S_UDT` and `S_CONSTANT` records; when there are no shared
symbols, that subsection is empty. It is followed, when the pack holds
function debug info, by a files subsection and a functions subsection, in that
order (see below). The pack id is BLAKE3-256 of the entire COFF
file, and the pack is stored as `<id>.cvp` with the id in lowercase hex.
Readers find packs by that name (see Finding packs). The distinct
extension keeps packs from being mistaken for, or globbed with, the objects of
a build: a pack passed to a link as an input would add its types as a module.

## Function debug info

The debug info of an inline function or a template is a COMDAT `.debug$S`
section associated with the function's COMDAT `.text` section, and every
object that emits the function repeats it. A pack can hold such sections.

The **functions subsection** of a pack is a `u32` count followed by one entry
per function: a `u32` size and that many bytes, padded with zeros to 4 bytes.
An entry is the section's content without its leading magic, with two kinds of
4-byte field rewritten in place:

- every type index of a symbol record that is not a simple type is an index
  in the pack's logical type stream;
- the `NameIndex` of every line block is an index into the pack's files.

The file operand of a `ChangeFile` annotation of an inline site is also an
index into the pack's files, written in the 4-byte form of compressed
annotation integers. Such an inline site is rebuilt: its annotations without
their padding, then zeros to a 4-byte boundary, and the record length and the
subsection size are updated. Restoring writes the operand in the fewest bytes,
as compilers do, so a pack entry and the section it restores can differ in
size.

The **files subsection** is a `u32` count followed by one entry per file:
`u32 name_size`, `u8 checksum_kind`, `u8 checksum_size`, `u16 reserved = 0`,
the name, the checksum, and zero padding to 4 bytes. A file is identified by
its name, checksum kind and checksum, as in an object's file checksums
subsection.

A packed object replaces the content of such a section with 24 bytes: the
magic `4` and one **function reference subsection** of 12 bytes:

```text
u32 slot           // 0
u32 ordinal        // function entry in that pack
u32 original_size  // size of the original section, magic included
```

Everything else about the section is kept: its header but for the raw-data
size, its relocations at their original offsets, and the COFF symbol table
entries, including the section definition's length and checksum. To restore
the section, a reader copies the entry after the magic and rewrites the
fields: a type index through the object's range map to the local index that
the map assigns to that pack index, and a file to the offset of the first
entry with the same identity in the object's file checksums subsection. The
restored section has `original_size` bytes. The object must have at most one
strings subsection and one file checksums subsection, in its non-COMDAT
`.debug$S` sections. Restoring needs no origin edit; the object's origin hash
covers the restored bytes. A linker restores the sections of live COMDATs
before it reads their symbols and lines.

Only sections that hold nothing but symbols and lines subsections can be
packed. Their symbols must all be known to type-index discovery, and must not
reference the string table: `S_FILESTATIC` and `S_DEFRANGE` are not packed.
The producer also keeps a section whose inline sites would not be restored to
the same bytes, such as annotations that are not minimally encoded.

## Origin record

`LF_CVPACK_ORIGIN` records the information needed to restore a rewritten
object. Its body is:

```text
u64 original_file_size
byte original_file_blake3[32]
u32 debug_h_header_size
u32 edit_count
byte debug_h_header[debug_h_header_size]
Edit edits[edit_count]
```

Each edit is:

```text
u32 section_index       // one-based COFF section index (bigobj files can
                        // have more than 65535 sections)
u16 kind                // 1 = type stream, 2 = symbol stream, 3 = .debug$H,
                        // 4 = directive, 5 = added directive section
u16 reserved = 0
u64 original_offset     // offset in the original section's raw data
u32 original_length
u32 packed_length
u32 restoration_size
byte restoration_data[restoration_size]
```

When `restoration_size` is nonzero, `restoration_data` contains:

```text
u32 fragment_count
Fragment fragments[fragment_count]
```

Each fragment is `u32 offset_in_original_span`, `u32 byte_count`, then
`byte bytes[byte_count]`. Fragments are ordered, nonempty, and nonoverlapping;
their offsets and lengths must fit in `original_length`. An edit with no
restoration fragments has `restoration_size = 0`.

Edits are ordered by section index and then original offset, and must not
overlap. Type and symbol map entries supply the removed records. The stored
whole-file BLAKE3 is the final materialization check.

- A type-stream edit covers the whole `.debug$T` section (offset 0).
- A `.debug$H` edit covers the whole section. The packed section keeps only
  its 8-byte header, which `debug_h_header` repeats. Materialization appends
  the BLAKE3 global hash of every restored type record, then applies the
  edit's restoration fragments, which hold the stored hashes that differ from
  the recomputed ones. The producer drops `.debug$H` when its header names
  BLAKE3 hashing, every record can be hashed, and at most 2000 hashes differ;
  otherwise the section is left unchanged.
- A symbol-stream edit covers one symbols subsection of a non-COMDAT
  `.debug$S` section, from its 8-byte subsection header to the end of its
  data. The packed subsection starts with its `S_CVPACK_MAP` records. Its
  length changes by a multiple of 4, so the padding after it is unchanged.
  Relocations of the section are moved with the inline records they apply
  to; packed records carry no relocations.
- A directive edit records the sentinel directive inserted into the object's
  `.drectve` section at `original_offset`, before any trailing NUL bytes:
  `original_length` is 0 and `packed_length` the directive's length.
  Materialization removes those bytes.
- An added directive section edit records a `.drectve` section that holds
  only the sentinel directive, added as the last section of an object that
  had none. `section_index` is that section, `original_length` is 0 and
  `packed_length` the directive's length. Materialization removes the section
  header and its data.
- `restoration_data` is produced only for `.debug$H` edits; readers reject it
  for other kinds.

Rewriting a section shifts every later file offset in the section table and
the COFF symbol-table pointer. An added section's header follows the last one,
and its data goes just before the symbol table. Both regular and bigobj
headers are supported.

## Packed-object recognition

A `.debug$T` stream with the signature `0x31505643` is packed, and must start
with `LF_CVPACK_LIST`. One with the signature `4` that starts with
`LF_CVPACK_LIST` is invalid. A consumer
must validate the list, optional origin, and maps before assigning type
indices. Treating these records as ordinary types changes subsequent indices
and can produce incorrect debug information.

## Interoperability

A packed object needs a linker that reads CodeView packs and finds its pack
(see Finding packs). Every packed object carries two safeguards for other
tools:

- **A sentinel directive.** Its `.drectve` contains
  `/INCLUDE:__cvpack_object_needs_lld_link_or_cvpack_materialize`. A linker
  that does not know packs cannot resolve that symbol, and fails with an error
  that names it, with or without debug information. LLD ignores it. An object
  has at most one `.drectve`: the directive goes into the existing one, or
  into a new last section.
- **A distinct signature.** Tools that read the object's CodeView themselves
  skip a `.debug$T` they do not recognize, instead of misreading its type
  indices.

`llvm-cvpack materialize` restores the original
object for any other tool, removing both. Measured with `link.exe` 14.52 and
LLD 23.1.3, which predates packs:

| Linker | Before the safeguards | With the safeguards |
|---|---|---|
| `link.exe /DEBUG` | `LNK1430` "bad relocation", or an internal error then `LNK1103` "debugging information corrupt; recompile module" | `LNK2001` unresolved external symbol `__cvpack_object_needs_lld_link_or_cvpack_materialize`, naming each object |
| `link.exe` without `/DEBUG` | links | the same `LNK2001` |
| LLD 23 `/DEBUG` (global hashes) | does not terminate | "undefined symbol: __cvpack_object_needs_lld_link_or_cvpack_materialize" |
| LLD 23 `/DEBUG:NOGHASH` | "The CodeView record is corrupted" | the same undefined symbol |
| LLD with packs, when the pack is not found | "missing CodeView pack … required by …" | unchanged |

Materialized objects link normally with both. An LLD that predates packs does
not terminate on a packed object without the sentinel: its global hashing
retries forever the records whose references never resolve. The sentinel stops
the link before type merging.

## Pack index

A pack's key index is stored next to it as `<id>.cvpx`, with the same id. It
lists the keys that a producer looks up when it packs objects against the pack,
so that it need not compute them from every record of the pack. Its layout is:

```text
byte magic[4] = "CVPX"
u16 version = 1
u16 flags = 0
byte pack_id[32]
u64 pack_file_size
u32 type_count            // records in the pack's logical type stream
u32 symbol_count          // S_UDT and S_CONSTANT symbols of the pack
u32 function_count        // function entries of the pack
u32 type_entries
u32 symbol_entries
u32 function_entries
Entry types[type_entries]
Entry symbols[symbol_entries]
Entry functions[function_entries]
```

Each entry is a 16-byte key and a `u32`: the zero-based index of a type record,
or the ordinal of a symbol or function. Each table is sorted by key and holds
each key once, with the first record that has it. The type table only holds
records that an object can refer to: those at the first index of their key
whose referenced records are held the same way. Keys are those defined under
Producing packs.

An index describes one pack: a reader ignores an index whose id, file size or
counts do not match the pack, or which it cannot read. Since a producer
verifies every object it packs by restoring it, it can load a pack described by
its index without checking the pack's content id.

## YAML

`obj2yaml` and `yaml2obj` read and write the CodeView pack records as structured
YAML: `CVPackList` (`PackId`), `CVPackOrigin` (`OriginalFileSize`,
`OriginalFileHash`, `DebugHHeader`, `Edits`) and `CVPackMap` (`Entries` of
`Skip`, `Slot`, `Count`, `FirstDelta`) for types, and `CVPackMap` for
`S_CVPACK_MAP`. Ids and hashes are 64 hex digits. Each YAML record is one
record in the object, so a round trip gives the same bytes. The function debug
info subsections are `!CVPackFunctionRef` (`Slot`, `Ordinal`,
`OriginalSize`), `!CVPackFiles` (`Files` of `FileName`, `Kind`, `Checksum`)
and `!CVPackFunctions` (`Functions`, one hex string per entry). A `.debug$S`
that holds only symbols or these subsections is written without a string
table.

## Finding packs

An object records the id of its pack and no path, so its bytes do not depend
on where it was packed. No option tells a reader where packs are: it looks for
`<id>.cvp` in directories it already knows.

| Reader | Directories, in the order searched |
|---|---|
| `lld-link` | the directory of the object, or of the archive that holds it; the directory of the output file; then the directories searched for a library: the current directory, `/libpath:` and `LIB` |
| `llvm-cvpack` | the directory of the object; then the `--pack-path` directories |

The first two directories of `lld-link` are those it searches for the PDB that
an `LF_TYPESERVER2` record names. A file found by its name cannot be another
pack: the name is the hash of the content, and the reader checks the content
against it. When no directory holds the pack, the error lists the directories
searched. `lld-link /reproduce:` stores the packs of a link in the reproducer.

`llvm-cvpack` writes a new pack to its output directory, next to the objects
packed against it, so a link of those objects needs no option. With
`--pack-path`, packs are in a directory of their own, which links name with
`/libpath:` or `LIB`.

## Producing packs

`llvm-cvpack pack --out-dir=<staging> <objects>`
reads all inputs and writes every input to the staging directory together with
`cvpacks.json`. Every packed object references one pack, the base. The pack
directories of a run are those of `--pack-path`, or the staging directory
without the option. The run writes a pack only when it makes a new base, into
the first pack directory. `--verbose` prints a summary and the reason each
unpacked input was left alone. Objects are read and written on all hardware
threads, or on `--threads=<n>`. See {doc}`CommandGuide/llvm-cvpack`.

The linker does not pack: it reads packs. A regular
`lld-link /debug ...` invocation reads packs while building
the PDB and accepts packed and ordinary objects together. It has no pack
option: it finds each pack by its name (see Finding packs). A link without
`/debug` does not read packs. To restore an object
for another linker or tool, or to check that a packed object and its pack are
intact, run
`llvm-cvpack materialize <packed.obj> -o <original.obj>`:
it restores every record and checks the result against the hash of the
original.

- **Keys.** A type record's key is BLAKE3-128 of the record with every type
  index replaced by the referenced record's global hash. `S_UDT` and
  `S_CONSTANT` symbols are keyed the same way. A COMDAT `.debug$S` section's
  key hashes the normalized section with markers in place of type and file
  references, followed by the referenced keys in field order. The same
  function key works across runs and can be stored in a pack index.
- **Machine.** A run packs the inputs of the machine of the most inputs, and
  stages the others unchanged.
- **Base.** The packs of the pack directories for that machine are the
  candidates; the base is the one that holds the most bytes of records the
  inputs contain.
- **Selection against the base.** A record moves out of an object when the
  base holds it at the first index of its key and the records it references
  are held the same way; global `S_UDT` and `S_CONSTANT` at scope depth 0,
  without relocations, and function debug info move when the base holds them.
  Everything else stays in the object. An object's packed form depends only on
  the object and the base: an unchanged object is staged with the same bytes,
  and so is a restored object packed against the base.
- **Rebase.** The run measures two quantities, and prints them with
  `--verbose`: the bytes that a rebase would move out of the objects beyond
  those the base moves (function debug info counts what it saves), and the
  bytes of the base that the inputs use, both against the bytes of the base's
  records. Restored objects count. When the first exceeds `<missing>`% of the
  base or the second falls below `<used>`%, the run starts over as a rebase;
  nothing was written before. The thresholds are set with
  `--auto-rebase=<missing>,<used>`, 25 and 50 by default;
  `--auto-rebase=no` never rebases. Between rebases, the objects keep at
  most `<missing>`% of the base's size in records a rebase would pack, and the
  base is at least `<used>`% used. Without a candidate pack, the run is a
  rebase, and `llvm-cvpack rebase` always is.
- **Selection in a rebase.** A record moves to the new base when its key
  occurs in at least two inputs, it is not a repeat within the object, and
  every record it references moves too: the packed records of an object are
  therefore closed under references and can be renumbered into the pack's
  index space. Symbols are selected the same way; a function's debug info
  moves when its key occurs in at least two inputs and every type it
  references moves. Sections that COFF symbols point into, and sections with a
  relocation over a rewritten field, stay in the object. `--verbose` reports,
  by reason, the function debug info that stays.
- **Packed inputs.** An input that is already packed, typically a cache hit
  restored on the producer, is staged unchanged when its pack is the base.
  Otherwise it is materialized and packed again. When it cannot be
  materialized, for example because its pack is neither next to it nor in a
  pack directory, it is staged as it is, with a warning.
- **Manifest.** `cvpacks.json` lists, for each staged object, the pack it
  needs, if any: a cache stores and restores it with the object. `packs` is
  the union of those lists,
  `new_packs` the pack written by this run, if any, and `base` the base's id.
  When an existing base was used, `base_bytes`, `base_used_bytes` and
  `missing_bytes` give the measurements; after an automatic rebase,
  `rebased_because` gives the reason.
- **Verification.** The producer materializes each object it rewrites and
  checks it against its original bytes before staging the packed form. An
  object that fails this check is staged unchanged and reported.

`--prune` deletes the packs of the first pack directory that no
staged object needs, such as the previous base after a rebase, with their
indexes. Use it only when that directory belongs to the build being packed:
objects of other builds may need the packs it deletes.

A run that writes a base also writes its index. A run that finds a pack without
a usable index builds the index, and keeps it when the pack is in the first
pack directory. `--base=<id>` selects one base candidate and
disables rebasing. That is the form for packing one object, or a few, at a
time. It requires the named `.cvp`. The `.cvpx` index speeds up packing, and
links and materialization only need the pack files.

### Packing a local build

A developer can pack their own build to save disk space, without a cache:

1. After a full build, `llvm-cvpack pack` over all the objects makes the base.
   Links use the staged objects, and find the base next to them.
2. After an incremental build, pack only the objects the build rewrote, with
   `--auto-rebase=no`. They are packed against the existing base, and no
   pack is written.
3. Occasionally, pack all the objects with the default thresholds, which
   rebases when needed, and use `--prune`.

Pass `--auto-rebase=no` whenever the inputs are a subset of the build. The
used measurement counts only the run's inputs, so a run over a few objects
finds the base mostly unused and would rebase: the new base would hold only
those objects' records.

A link accepts packed and unpacked objects together, so step 2 is optional: it
can wait until the unpacked objects reach some share of the build. When packed
objects replace the build's own outputs, check that the build system does not
treat a rewritten output as out of date.

## Integration

Packing is a pure function of an object and a base, and its output restores
the object exactly. Where a build system calls it depends on who writes its
cache.

### Where packing can happen

| Model | Who writes the cache | Where packing happens | Build graph change |
|---|---|---|---|
| A. Host-populated cache | the machine that ran the build | one pack step over the build's objects, before publishing | none |
| B. Pack in the compile action | whoever runs the action, local or remote | a compiler wrapper: compile, then pack against a pinned base | the wrapper replaces the compiler |
| C. Separate pack action | whoever runs the action, local or remote | an action `pack(object, base)` between compile and link | one more action per object |
| D. In the cache store | unchanged | the cache client or server packs objects it stores and restores them when it reads them | none |

- **A** is the flow described under Producing packs, and the one measured
  below. The cache key stays the compile's key; only the stored value is
  packed. A restore fetches the packs that `cvpacks.json` lists for the
  object. It fits caches written by the machine that drives the build, and
  local caches.
- **B** needs no change to the build graph, but the base becomes an input of
  every compile action: a new base invalidates every cached compile. It suits
  a base that changes as rarely as the toolchain.
- **C** keeps compiles cached as they are. The pack action's key covers the
  object and the base, so a new base only invalidates pack actions, which are
  cheap. Links take the base as an input.
- **D** is invisible to the build: digests, action keys and outputs do not
  change, much like a store that compresses blobs. A cache can pack each object
  as it arrives and materialize on reads. Store-side packing saves backend
  storage. A client or proxy that can exchange packed objects with the store
  can also save transfer on that connection. Local disks and links see
  original objects.

B, C and D pack one object at a time: see Packing one object at a time below.

### Remote execution caches

With a remote execution protocol such as REAPI (Bazel, Siso), remote workers
write the outputs of the actions they run, and the action cache records them.
A client should not replace those outputs with packed objects, even where the
server accepts client writes:

- the cache entry would no longer be what the action produces, and clients
  that do not use packs would receive objects they cannot link;
- a link action's key covers the digests of its inputs, so clients with packed
  objects and clients with original objects would no longer share link cache
  hits;
- many deployments only accept writes from trusted workers.

Packed objects must instead come from an action (B or C), or stay invisible to
the protocol (D).

### Examples

- **Unreal Engine.** With a cache written by the machine that drives the
  build, UnrealBuildTool, or a step after it, packs the build's objects before
  publishing them (A). Its links need no option when the packs are next to the
  objects, and `/libpath:` when they are in a directory of their own.
  Hot-reload tools that read objects need materialized objects or unpacked
  builds.
- **FASTBuild.** If the cache is written by the machine that drives the build,
  this is A again. A wrapper compiler (B) can also carry the base as one of the
  files sent to workers, which makes the base part of the toolchain's identity.
- **Chromium.** GN, Siso and remote execution: C puts packed objects on
  developers' disks and shrinks link inputs; D only shrinks the cache.
- **Local builds without a cache:** see Packing a local build.

### Packing one object at a time

`llvm-cvpack pack --base=<id> --pack-path=<dir> --out-dir=<dir> <object>`
packs one object against a base through the base's index (see Pack index),
instead of computing the keys of every record of the base. The decision is the
same as in a batch run, and so are the bytes: a record moves out when the base
holds it at the first index of its key, and the records it references move
too. On the Unreal Engine client's base (540 MB, 5.5 million records, a 121 MB
index), packing one object takes about 0.6 s plus 0.03 s per MB of object:

| Object | Before the index | With the index |
|---|---:|---:|
| 0.6 MB, the 10th percentile | — | 0.65 s |
| 11.5 MB, the median | — | 0.94 s |
| 24.8 MB | 5.75 s | 1.3 s |
| 34.3 MB, the 90th percentile | — | 1.55 s |

What remains of the fixed cost is reading the index and parsing the pack's
records; an index that also held the offsets of the pack's records would avoid
the second. The current producer loads both files, uses the index to select
records, and materializes the rewritten object against the pack to verify it.
The compiler writes complete objects, which are packed afterwards. Direct
compiler emission would make the base an input of each compile.

A batch pack step makes the base by looking for shared records across its
inputs. Per-object packing then uses a base made from earlier objects.

1. **Bootstrap.** Start with ordinary objects. A recent build with the same
   toolchain, target family and debug settings is a starting corpus. Its
   objects can come from a local build, CI or a cache that can materialize
   them. The current batch command writes the base and its index and stages
   the inputs. A cache could publish just the base and index for future
   uploads. A training-only mode would avoid staging.
2. **Current base.** A setting names the current base by its content id; see
   The current base setting below for where it lives.
3. **Every build.** A wrapper, pack action or cache uploader can pack each
   completed object independently against the selected base. Records the base
   lacks stay in the object. A caller that cannot fetch the base can use the
   ordinary object.
4. **Rebase.** A batch run over a representative corpus can measure how much
   is missing from the base and how much of it is used (see Rebase). To decide
   whether to roll the pin, compare the old and new bases on a later build and
   include the cost of fetching the base. Publish the new base and index
   before updating the setting. Running builds can finish with the old base.
5. **Old bases** stay in the store as long as a stored object needs them. A
   link may see objects packed against several bases, and reads every base
   they reference.

### The current base setting

Where the setting lives depends on whether packing changes the output of a
build action. For B and C, the pin is a declared action input and can be
versioned with the source. For A and D, base selection is a cache policy. A
project can check in the pin and pass it to the uploader separately from the
compile command, or the cache operator can manage it.

| Scenario | Where the setting lives | Who updates it |
|---|---|---|
| A. Host-populated cache, batch pack step | an alias in the cache store, such as `cvpack/<branch>/current`, naming the pack to download into the pack directory; the pack step then picks the base among the packs it finds | the host or CI, after a pack step that rebased |
| B. Pack in the compile action | checked into the source tree, next to the toolchain pin; with FASTBuild, the base file listed among the wrapper compiler's extra files, which makes it part of the toolchain's identity | a commit, which an automated roll can make |
| C. Separate pack action | a checked-in file or dependency declaration with the base's BLAKE3-256 ID. A downloader such as Bazel's `http_file` can separately verify the download with its SHA-256 | a commit, which an automated roll can make |
| D. In the cache store | cache configuration or a checked-in pin passed as upload policy | the cache operator or a commit |
| Local builds without a cache | nothing: the pack directory holds a single base, kept so by `--prune` | the developer, when packing all objects again |

- **Read dependencies.** Each object's `LF_CVPACK_LIST` names the pack it
  needs. A cache can copy the dependency list from `cvpacks.json` into the
  entry's metadata and start fetching missing packs alongside the packed
  object. Restores and links use those exact IDs.
- **A stale setting is never wrong.** Objects packed against an older base
  restore and link exactly; they only keep more records inline. A mutable alias
  is therefore safe for A and D.
- **Scope it per branch, or per family of targets.** Branches drift apart, and
  so may unrelated targets. One base per branch is the simplest start.
- **B and C pin it for reproducibility, not for correctness:** a checkout must
  always give the same action keys, which a mutable alias cannot guarantee.
- **The build scripts can pass it on the command line,**
  `--base=<content id>`, read from the pinned value. A remote execution
  action key covers the command line and the digests of the declared inputs,
  the base and its index, so a new base changes the key of every pack action
  that uses it. An explicit base is also never replaced by the run: the pack
  step neither picks among the packs it finds nor rebases.

### Compared with precompiled headers

Packing in the compiler, against a base, would resemble a precompiled header:
the compiler reads a prebuilt artifact made from content that many translation
units share. They differ where it matters for builds:

| | Precompiled header | Base pack in the compiler |
|---|---|---|
| What is shared | the work: parsing and semantic analysis | the output: debug records |
| Saves | compile time | object size, transfer, storage and link inputs |
| Built from | a header set chosen before the build | the records that the objects of an earlier build share |
| Granularity | one per translation unit, with the same include prefix and options | any record from any header, in any include order; a translation unit uses what matches |
| When out of date | an error, or every translation unit that uses it is compiled again | still correct: the records it lacks stay in the objects |
| When absent | the compile cannot run | the object is written unpacked |
| In the build graph | a dependency built first: editing one of its headers compiles every translation unit that uses it again | an optional input |

The two combine: precompiled headers make compiling faster, and a base makes
its output smaller. Skipping code generation for inline functions that the
base already holds would move packs from sharing output to sharing work, with
a correctness condition like that of a precompiled header: the compiler would
have to know that the base's copy is what it would emit.

## Measurements

These figures were measured on three code bases while the format was
developed. "zstd-18" compresses each object and the pack separately at zstd
level 18, as a cache stores them. Unless noted, links use `/DEBUG:GHASH`.

- **clang**: the 3,278 objects of `clang.exe`, built with clang-cl.
- **Unreal Engine project**: a game built with unity files and a clang-cl 20
  toolchain. Its targets are the game client ("UE client"), the editor ("UE
  editor") and ShaderCompileWorker.
- **Chromium**: a GN debug build (`is_debug=true`, `symbol_level=2`,
  `-gcodeview-ghash`), built with a toolchain from this branch.

### Size

| Inputs | Original objects | Original objects, zstd-18 | CodeView packs | CodeView packs, zstd-18 |
|---|---:|---:|---:|---:|
| clang.exe | 3.69 GB | 0.74 GB | 1.96 GB | 0.38 GB (types only) |
| UE client Development | 39.47 GB | 7.99 GB | 7.96 GB | 0.98 GB |
| UE client DebugGame | 42.56 GB | 8.28 GB | 10.62 GB | 1.21 GB |
| UE editor Development | 48.16 GB | 9.69 GB | 9.87 GB | 1.23 GB |
| UE editor DebugGame | 28.75 GB | 5.54 GB | 6.05 GB | 0.61 GB |
| UE client Shipping, ThinLTO cache | 28.87 GB | 6.02 GB | 5.18 GB | 0.62 GB |
| ShaderCompileWorker Development | 1.39 GB | 0.28 GB | 0.59 GB | 0.09 GB |
| Chromium, component build (48,522 objects) | 65.9 GB | 11.87 GB | 30.5 GB | 4.08 GB |
| Chromium `chrome.dll`, non-component (46,379 objects) | 34.65 GB | — | 24.1 GB | — |

"Original objects" are the compiler's output. "CodeView packs" is the total
of the same objects packed against one base and of that base. "clang.exe,
CodeView packs, zstd-18" was measured before function debug info was packed.

- **Function debug info.** Moving COMDAT `.debug$S` into the pack saves
  0.82–1.24 GB raw per Unreal Engine target, 0.23 GB on clang and 6.58 GB on
  the Chromium component build. After zstd-18 it adds only 4–5% over type
  packing, because the copies compress well. In the ThinLTO cache it moves 9 MB:
  ThinLTO already deduplicates inline functions before code generation.
- **What stays in packed objects** (UE client DebugGame): relocations
  (1.28 GB), the COFF string table (1.64 GB, 37% duplicate names), types
  unique to one translation unit (1.45 GB), the symbol table (1 GB), section
  headers (0.9 GB), and functions that exist in only one object (0.7–2.3 GB per
  target). Identical code in repeated COMDATs is about 370 MB, and is not debug
  information.
- **Range maps.** clang's objects need 3.97 M map entries in 21.9 MB, 4.9
  types per entry. Laying out the pack so that records used by the same
  objects are adjacent, to lengthen the runs, removed under 4% of the entries
  and was dropped.

### Objects against the linked output

How much a link reads, compared with what it writes:

| Link | Output | Original objects | CodeView packs |
|---|---:|---:|---:|
| Chromium `chrome.dll`, component build | 5.19 GB (DLL 0.43 GB + PDB 4.76 GB) | 24.18 GB, 4.7x the output | 13.75 GB, 2.6x the output |
| clang.exe | 1.08 GB (PDB only) | 3.69 GB, 3.4x the PDB | 1.96 GB, 1.8x the PDB |

For `chrome.dll`, "original objects" are the 17,268 objects the link loads,
and "CodeView packs" is their packed form (12.36 GB) plus the pack of the
whole build (1.39 GB), of which the link reads only the records it uses.

The remaining factor is not debug information that could still be shared. On
clang, the unique type bytes of the packed objects and the pack together
(231 MB) already match the type streams of `clang.pdb` (TPI and IPI, 222 MB).
What else the objects hold is either unique to one translation unit, or data
that a PDB never stores, such as relocations, COFF symbol and string tables,
section headers, and code and data.

### Compression

The 3,278 objects of `clang.exe` (3.69 GB, the first row of the size table),
compressed one object at a time at each zstd level. The CodeView packs
columns were measured before function debug info was packed. "+ 1 MB
dictionary" sizes include the dictionary.

| zstd level | Original objects | Original objects + 1 MB dictionary | CodeView packs | CodeView packs + 1 MB dictionary |
|---:|---:|---:|---:|---:|
| none | 3,694.6 MB | — | 2,154.1 MB | — |
| 3 | 890.5 MB | 831.0 MB | 468.3 MB | 449.3 MB |
| 9 | 815.2 MB | 727.7 MB | 426.6 MB | 395.8 MB |
| 15 | 804.8 MB | 718.0 MB | 419.1 MB | 390.3 MB |
| 18 | 742.9 MB | 657.3 MB | 377.4 MB | 349.6 MB |
| 19 | 739.5 MB | 654.6 MB | 375.5 MB | 348.4 MB |

A dictionary trained on the build complements packing but does not replace
it: it saves 12% at level 18, packing saves 49%, both together 53%. Level 18
costs about 3x the CPU of level 15 for 8% less. Compressing COFF string tables
with a mangling-aware shared table would save at most part of their 19 MB
zstd-18 size, so it was not pursued.

### Pack step

| Inputs | Time | Peak memory |
|---|---:|---:|
| clang.exe | 9.6 s | 4.6 GB |
| Unreal Engine targets | 46–81 s | 6.2–8.1 GB |
| Chromium component build | 101 s | 17.6 GB |
| Chromium `chrome.dll`, non-component | 58 s | — |

The times include verifying every staged object by restoring it. They were
measured when the producer was a mode of `lld-link`; `llvm-cvpack` runs the
same steps. Storing global hashes in the base cuts the hashing in a link from
3.2 s to 0.8 s on clang.

### Link output

A link of packed objects gives the same image and PDB as a link of the
originals:

- **clang.exe:** byte-identical `.exe` and `.pdb`.
- **Unreal Engine project:** identical but for the PDB identity, the DBI
  stream (object paths) and the linker's module, for example 4,539 of 4,542
  PDB streams on the UE client Development, linked without `/DEBUG:GHASH`.
- **Chromium:** the same, with 17,638 of 17,641 streams identical for
  `chrome.dll`. The DLLs differ only in the 8 bytes of the PDB GUID.

Rebuilding a static library, even from its original members, changes the
order in which LLD loads them and so the type order. Packed links are
therefore compared with links of libraries rebuilt from the original members.

### Link time and memory

| Link | Time, original → packed | Peak working set | Peak private |
|---|---|---|---|
| clang.exe | 6.51 → 7.07 s | — | — |
| UE client Development | 54.5 → 55.1 s | — | — |
| Chromium `blink_platform.dll` | 7.7 → 9.8 s | 5.9 → 6.8 GB | 2.5 → 3.2 GB |
| Chromium `chrome.dll`, component | 36.5 → 36.2 s | 30.5 → 24.9 GB | 9.4–10.3 → 11.0 GB |
| Chromium `chrome.dll`, non-component | 79 → 78 s | 51.4 → 46.3 GB | 19.9–21.3 → 21.9 GB |

The private ranges are run-to-run variation of the baseline. A packed link
must find the staged objects in the file cache: a cold run of the
non-component link took 110 s.

The packed objects carry no `.debug$H`, so LLD computes hashes that a
`-gcodeview-ghash` baseline maps from its inputs. For `chrome.dll` (component
build, 128 M logical type records) the per-object state was first 36 bytes per
logical record, 6.5 GB above the baseline. LLD now keeps:

- each object's logical type stream as runs of inline records and of pack
  ranges (0.29 GB instead of 3.57 GB);
- hashes only for inline records, using the pack's hashes for the others
  (0.09 GB instead of 1.03 GB);
- a global hash table sized by the records actually inserted, inline records
  plus used pack records, with a quarter of headroom (0.15 GB instead of
  1.03 GB, also below the baseline's).

What remains above the baseline is mostly restored function debug info
(0.26 GB on the component `chrome.dll`, 0.63 GB on the non-component one), the
pack index maps (0.21–0.31 GB) and arrays sized by the whole pack rather than
the records a link uses, about 360 MB for a 10 M-record pack. That fixed cost
dominates small links such as `blink_platform.dll`.

A non-component debug `chrome.dll` cannot be linked with or without packs: its
public and global symbol records (2.19 + 2.21 GB) exceed the 4 GB a PDB's
symbol record stream can hold. The figures above are for the link up to the
point where it writes the PDB.

Linking each static library's members into a prelinked type stream first was
lossless on clang, but slower (9.3 s against 6.5 s) and used more private
memory (2.05 GB against 1.2 GB).

### Keeping one base across builds

A cache stores each distinct object once, so what a later build costs is the
bytes of its objects that changed, plus any new pack. Records that are new
since the base stay inline in the objects that use them. Once too many are
missing from the base, the producer makes a new one (see Rebase above).

A change spreads further than it looks: a record's key covers the records it
references, so changing one type changes the key of every type that uses it,
directly or not. How long a base stays useful therefore depends on the code
base. Measured over four builds of the UE client Development, one day, one
week and one month apart, at zstd-18:

| New bytes stored | Full rebase every build | Single base, automatic rebase (default) | Chain of delta packs (dropped) |
|---|---:|---:|---:|
| Base build | 968 MB | 968 MB | 968 MB |
| +1 day | 966 MB | 221 MB | 264 MB |
| +1 week | 967 MB | 928 MB | 926 MB |
| +1 month | 975 MB | 975 MB (rebased) | 915 MB |
| Total for the 3 later builds | 2,908 MB | 2,124 MB | 2,105 MB |
| Packs an object needs after a month | 1 | 1 | 4 |

- **A full rebase every build** stores about 40% more: every object changes,
  even when its source did not.
- **The single base** stays relevant for weeks on this project. A rebase
  would have packed only 1%, 12% and 37% more than the base held; the default
  threshold of 25% triggered a rebase at the month step only.
- **A chain**, in which each build writes a small delta pack holding its new
  shared records on top of the previous packs, stores about the same. It costs
  more: each object needs every pack of the chain, and the chain grows with
  every build. It was dropped, and packs no longer refer to other packs.

On upstream LLVM, where a daily sync recompiles every object, the base goes
stale within a day: a rebase would have packed 260% of the base's size beyond
it. The default then rebases at every step, which stores as much as a full
rebase. A chain stored 20% less after one day, and 2–3% less after a week or
a month.

### Where the debug info comes from

Share of the type bytes and of the function debug info bytes, by the header
that defines them. "Project" includes Unreal Engine itself and LLVM itself.

| Code base | Origin | Type bytes | Function debug info bytes |
|---|---|---:|---:|
| clang, objects | MSVC headers | 20% | 9% |
| | Project | 74% | 91% |
| UE client Development, objects | MSVC headers | 11% | 8% |
| | Windows SDK | 0.5% | 0 |
| | Third-party | 0.9% | 1.2% |
| | Project | 85% | 90% |
| UE client Development, base pack | MSVC headers | 13% | 6% |
| | Windows SDK | 1.3% | 0 |
| | Third-party | 5% | 0.6% |
| | Project | 78% | 93% |
| Chromium, objects | libc++ | 30% | 39% |
| | MSVC headers | 3.6% | — |
| | Windows SDK | 0.1% | — |
| | Blink | 17% | — |
| | V8 | 12% | — |
| | `base` | 12% | — |
| Chromium, base pack | libc++ | 39% | 48% |

System headers are shared by most objects, so the pack stores them once:
about 60 MB of the Unreal Engine project's 440 MB base. They add no churn
while the toolchain is unchanged, except templates instantiated over project
types, such as `std::vector<T>` of a project type.

### What gains are left

Sharing debug information is close to complete: the shared types and function
debug info are stored once, and what stays in the objects is unique to one
translation unit. A pack is a COFF file, though, and could hold any content
that objects repeat. Measured candidates:

| Content | Repeated content | Notes |
|---|---|---|
| Relocations of packed function debug info | about 190 MB (UE client DebugGame) | They stay in the objects at their original offsets; the pack could describe them, as they are the same in every copy but for symbol indices. |
| Code and unwind data of repeated COMDATs (`.text`, `.pdata`, `.xdata`) | about 370 MB (UE client DebugGame), about 66 MB (clang) | Objects would keep a reference, as for function debug info, and LLD would restore only the copies it keeps. |
| COFF symbol names | 1.64 GB string table, 37% duplicates (UE client DebugGame); 134.6 MB, 64.5 MB distinct (clang) | zstd already compresses names 7x per object, so the gain after compression is small. |
| Section headers and symbol table entries | about 0.9 GB and 1 GB (UE client DebugGame) | Inherent to one COMDAT section per function. |

Two directions go beyond object size:

- **Link time from the pack.** Pack records are already unique and
  content-addressed, so a link that did not have to reproduce the original
  type order could emit them as one block of the PDB's type stream, without
  hashing or merging them.
- **Packing at compile time.** A compiler that packs against the base as it
  writes the object would make remote compiles return packed objects, saving
  the transfer as well as the cache storage.
