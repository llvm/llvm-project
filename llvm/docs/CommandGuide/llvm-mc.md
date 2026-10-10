# llvm-mc - LLVM Machine Code Playground

:::{program} llvm-mc
:::

## SYNOPSIS

{program}`llvm-mc` \[*options*\] \[*filename*\]

## DESCRIPTION

The {program}`llvm-mc` command takes assembly code for a specified architecture
as input and generates an object file or executable.

{program}`llvm-mc` provides a set of tools for working with machine code,
such as encoding instructions and displaying internal representations,
disassembling strings to bytes, etc.

The choice of architecture for the output assembly code is automatically
determined from the input file, unless the {option}`--arch` option is used to
override the default.

## OPTIONS

If the {option}`-o` option is omitted, then {program}`llvm-mc` will send its
output to standard output if the input is from standard input. If the
{option}`-o` option specifies "`-`", then the output will also be sent to
standard output.

If no {option}`-o` option is specified and an input file other than "`-`" is
specified, then {program}`llvm-mc` creates the output filename by taking the
input filename, removing any existing `.s` extension, and adding a `.o`
suffix.

Other {program}`llvm-mc` options are described below.

### End-user Options

:::{option} --help
Display available options (--help-hidden for more).
:::

:::{option} -o <filename>
Use `<filename>` as the output filename. See the summary above for more
details.
:::

:::{option} --arch=<string>
Target arch to assemble for, see -version for available targets.
:::

:::{option} --as-lex
Apply the assemblers "lexer" to break the input into tokens and print each of
them out. This is intended to help develop and test an assembler
implementation.
:::

:::{option} --assemble
Assemble assembly file (default), and print the result to assembly. This is
useful to design and test instruction parsers, and can be a useful tool when
combined with other llvm-mc flags. For example, this option may be useful to
transcode assembly from different dialects, e.g. on Intel where you can use
-output-asm-variant=1 to translate from AT&T to Intel assembly syntax. It can
also be combined with --show-encoding to understand how instructions are
encoded.
:::

:::{option} --disassemble
Parse a series of hex bytes, and print the result out as assembly syntax.
:::

:::{option} --mdis
Marked up disassembly of string of hex bytes.
:::

:::{option} --cdis
Colored disassembly of string of hex bytes.
:::

:::{option} --filetype=[asm,null,obj]
Sets the output filetype. Setting this flag to `asm` will make the tool output
text assembly. Setting this flag to `obj` will make the tool output an object
file. Setting it to `null` causes no output to be created and can be used for
timing purposes. The default value is `asm`.
:::

:::{option} -g
Generate DWARF debugging info for assembly source files.
:::

:::{option} --hex
Take raw hexadecimal bytes as input for disassembly. Whitespace is ignored.
:::

:::{option} --large-code-model
Create CFI directives that assume the code might be more than 2 GB.
:::

:::{option} --main-file-name=<string>
Specify the name we should consider the input file.
:::

:::{option} --masm-hexfloats
Enable MASM-style hex float initializers (3F800000r).
:::

:::{option} -mattr=a1,+a2,-a3,... Target specific attributes (-mattr=help for details).
:::

:::{option} --mcpu=<cpu-name>
Target a specific cpu type (-mcpu=help for details).
:::

:::{option} --triple=<string>
Target triple to assemble for, see -version for available targets.
:::

:::{option} --split-dwarf-file=<filename>
DWO output filename.
:::

:::{option} --show-inst-operands
Show instructions operands as parsed.
:::

:::{option} --show-inst
Show internal instruction representation.
:::

:::{option} --show-encoding
Show instruction encodings.
:::

:::{option} --save-temp-labels
Don't discard temporary labels.
:::

:::{option} --relax-relocations
Emit R_X86_64_GOTPCRELX instead of R_X86_64_GOTPCREL.
:::

:::{option} --print-imm-hex
Prefer hex format for immediate values.
:::

:::{option} --preserve-comments
Preserve Comments in outputted assembly.
:::

:::{option} --output-asm-variant=<uint>
Syntax variant to use for output printing. For example, on x86 targets
--output-asm-variant=0 prints in AT&T syntax, and --output-asm-variant=1
prints in Intel/MASM syntax.
:::

:::{option} --compress-debug-sections=[none|zlib|zstd]
Choose DWARF debug sections compression.
:::

:::{option} --verify=<prefixes>

Instead of just printing diagnostics, check them against `<prefix>-error`,
`<prefix>-warning`, `<prefix>-note` and `<prefix>-remark` comments in
the input, similar to Clang's `-verify` flag. `<prefixes>` is a
comma-separated list of prefixes to recognize; a bare `--verify` (with no
`=<prefixes>`) is equivalent to `--verify=expected`.

A directive has the form:

``asm
<prefix>-<kind>[-re][@<offset>][:<col>] {{<text>}}
``

`<kind>` is one of `error`, `warning`, `note` or `remark`. Without
an `@<offset>`, the directive applies to its own line; `@+N`/`@-N`
apply to the line `N` lines below/above, and `@above`/`@below` apply to
the nearest non-directive line above/below. An optional `:<col>` (matching
the column shown in the tool's own `file:line:col:` diagnostic output)
additionally requires the diagnostic to be at that exact column; without it,
only the line is checked. By default `<text>` is matched as a substring of
the diagnostic's message; with the `-re` suffix, any `{{...}}` block
nested inside `<text>` is matched as a regular expression instead (the
rest of `<text>` is matched literally), for example::

```asm
## Matches only diagnostics containing this exact text, on this line.
.foo
# expected-error@-1 {{unknown directive}}

## Also requires the diagnostic to be at column 1.
.foo
# expected-error@-1:1 {{unknown directive}}

## Matches any diagnostic starting with "unknown" and ending with
## "directive"; only the nested {{.*}} is a regex, the surrounding text
## is literal.
.foo
# expected-error-re@-1 {{unknown {{.*}} directive}}
```

A directive must appear within a comment recognized by the target being
assembled for (its own comment string, plus `//` and `#` if the target
allows additional comment forms); text that merely resembles a directive
elsewhere on the line is ignored. Directives are matched line-based, so they
are not recognized inside multi-line `/* ... */` block comments. Use a
single-line comment form instead.

Under `--verify`, the process exit code reflects whether the diagnostics
matched what was expected, not whether assembling/disassembling itself
produced an error: a fully expected error is a successful (exit code 0) run.
`--verify` is not supported together with `--as-lex`, since `--as-lex`
does not report its errors as diagnostics with a message or location.

:::

## EXIT STATUS

If {program}`llvm-mc` succeeds, it will exit with 0. Otherwise, if an error
occurs, it will exit with a non-zero value.

