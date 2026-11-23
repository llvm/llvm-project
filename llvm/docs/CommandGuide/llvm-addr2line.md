# llvm-addr2line - a drop-in replacement for addr2line

```{program} llvm-addr2line
```

## SYNOPSIS

{program}`llvm-addr2line` \[*options*\]

## DESCRIPTION

{program}`llvm-addr2line` is an alias for the {manpage}`llvm-symbolizer(1)`
tool with different defaults. The goal is to make it a drop-in replacement for
GNU's {program}`addr2line`.

Here are some of those differences:

- `llvm-addr2line` interprets all addresses as hexadecimal and ignores an
  optional `0x` prefix, whereas `llvm-symbolizer` attempts to determine
  the base from the literal's prefix and defaults to decimal if there is no
  prefix.
- `llvm-addr2line` defaults not to print function names. Use `-f` to enable
  that.
- `llvm-addr2line` defaults not to demangle function names. Use `-C` to
  switch the demangling on.
- `llvm-addr2line` defaults not to print inlined frames. Use `-i` to show
  inlined frames for a source code location in an inlined function.
- `llvm-addr2line` uses `--output-style=GNU` by default.
- `llvm-addr2line` parses options from the environment variable
  `LLVM_ADDR2LINE_OPTS` instead of from `LLVM_SYMBOLIZER_OPTS`.
- `llvm-addr2line` accepts an address with a '+' prefix, e.g. `+0x00777fff`.
  This is treated as a symbol name by `llvm-symbolizer`.

## SEE ALSO

{manpage}`llvm-symbolizer(1)`

[--output-style=gnu]: llvm-symbolizer.html#llvm-symbolizer-opt-output-style
[-c]: llvm-symbolizer.html#llvm-symbolizer-opt-c
[-f]: llvm-symbolizer.html#llvm-symbolizer-opt-f
[-i]: llvm-symbolizer.html#llvm-symbolizer-opt-i
