# llvm-addr2line - a drop-in replacement for addr2line

## SYNOPSIS

**llvm-addr2line** [*options*]

## DESCRIPTION

**llvm-addr2line** is an alias for the [llvm-symbolizer](llvm-symbolizer) tool
with different defaults. The goal is to make it a drop-in replacement for
GNU's **addr2line**.

Here are some of those differences:

* Defaults not to print function names. Use [-f](llvm-symbolizer-opt-f)
  to enable that.

* Defaults not to demangle function names. Use [-C](llvm-symbolizer-opt-C)
  to switch the demangling on.

* Defaults not to print inlined frames. Use [-i](llvm-symbolizer-opt-i)
  to show inlined frames for a source code location in an inlined function.

* Uses [--output-style=GNU](llvm-symbolizer-opt-output-style) by default.

## SEE ALSO

Refer to [llvm-symbolizer](llvm-symbolizer) for additional information.
