# llvm-readelf - a drop-in replacement for readelf

## SYNOPSIS

**llvm-readelf** [*options*]

## DESCRIPTION

**llvm-readelf** is an alias for the [llvm-readobj](llvm-readobj.html) tool with
a command-line interface and output style more closely resembling GNU
**readelf**.

Here are some of those differences:

* Uses `--elf-output-style=GNU` by default.

* Allows single-letter grouped flags (e.g. `llvm-readelf -SW` is the same as
  `llvm-readelf -S -W`).

* Allows use of `-s` as an alias for `--symbols` (versus `--section-headers` in
  **llvm-readobj**) for GNU **readelf** compatibility.

* Prevents use of `-sr`, `-sd`, `-st` and `-dt` **llvm-readobj** aliases, to
  avoid conflicting with standard GNU **readelf** grouped flags.

## SEE ALSO

Refer to [llvm-readobj](llvm-readobj.html) for additional information.
