# llvm-cvpack - CodeView pack tool

```{program} llvm-cvpack
```

## SYNOPSIS

{program}`llvm-cvpack` *command* \[*options*\] *inputs...*

## DESCRIPTION

The {program}`llvm-cvpack` tool moves the CodeView debug information that COFF
objects share to a CodeView pack, restores packed objects, and prints packs.
The format is experimental; {doc}`/CodeViewPacks` specifies it and describes
how a build can use it.

A packed object names the pack it refers to and carries a record of what was
taken out of it, so that it can be restored byte for byte. `lld-link` links
packed objects directly. Other linkers and tools need the restored object.

An object names its pack by content id, as `<content id>.cvp`, and records no
path. {program}`llvm-cvpack` searches a pack in the directory of the object
that names it, then in the {option}`--pack-path` directories. `lld-link`
searches it next to the object or its archive, next to the output file, then
where it searches for a library: in the current directory, and in the
directories of `/libpath:` and `LIB`.

## COMMANDS

`pack`
: Pack objects against the base pack of the pack directories: those of
  {option}`--pack-path`, or the output directory. Without a pack,
  or when the base no longer fits the objects, write a new base from what the
  objects share.

`rebase`
: Pack objects against a new base pack, written from what the objects share,
  without reading the packs of the pack directories.

`materialize`
: Restore the original bytes of a packed object.

`dump`
: Print the records of packs.

## PACK AND REBASE OPTIONS

{program}`llvm-cvpack` `pack` \[*options*\] *objects...*

{program}`llvm-cvpack` `rebase` \[*options*\] *objects...*

Every input is written to the output directory under its own file name: packed
when it shares records with the base, and unchanged otherwise. An input that is
not a COFF object with a CodeView type stream is copied. Each packed object is
checked to restore to its input before it is written.

The output directory also receives `cvpacks.json`, which lists, for each
object, the pack that it needs. A new pack, `<content id>.cvp`, and its key
index, `<content id>.cvpx`, are written to the first pack directory.

:::{option} --pack-path=<dir>
A pack directory. The option can be given several times. Without it, the
output directory is the pack directory: a new pack is then next to the objects
packed against it, where a link finds it without an option. With it, links
name the directory with `/libpath:` or `LIB`.
:::

:::{option} --out-dir=<dir>
The directory that receives the objects and `cvpacks.json`, and a new pack
unless {option}`--pack-path` is given. Required.
:::

:::{option} --base=<id>
Pack against the pack with this content id, given as 64 hexadecimal digits,
and never rebase. This is the form for packing one object, or a few, at a
time. `pack` only.
:::

:::{option} --auto-rebase=<missing>,<used>
Start over as a rebase when the objects hold more than *missing* percent of
the size of the base in records that a rebase would pack, or use less than
*used* percent of the base. The default is `25,50`. `--auto-rebase=no` never
rebases. `pack` only.
:::

:::{option} --prune
Delete the packs of the first pack directory that no object of this run needs,
with their indexes. Use it only when that directory belongs to the build being
packed.
:::

:::{option} --threads=<n>
The number of threads that read and write objects. The default is the number
of hardware threads.
:::

:::{option} --verbose
Print the phases of the run, a summary, and why objects were left unchanged.
:::

## MATERIALIZE OPTIONS

{program}`llvm-cvpack` `materialize` \[{option}`--pack-path`=*dir*\] *object*
{option}`-o` *file*

:::{option} -o <file>
The restored object. `-` writes it to the standard output. The output can be
the input. Required.
:::

The pack that the object names is searched next to the object, then in the
{option}`--pack-path` directories. The restored object is checked against the
hash of the original that the packed object records.

## DUMP OPTIONS

{program}`llvm-cvpack` `dump` *packs...*

Prints the content id of each pack, the machine of its objects, and its type
and symbol records.

## EXIT STATUS

{program}`llvm-cvpack` returns 1 on a usage error, when a pack or an output
cannot be read or written, and when `materialize` or `dump` cannot process its
input. Otherwise it returns 0. With `pack` and `rebase`, an object that cannot
be packed is not an error: it is copied unchanged, and reported with a warning
or with {option}`--verbose`.

## SEE ALSO

{manpage}`llvm-pdbutil(1)`, {manpage}`llvm-readobj(1)`
