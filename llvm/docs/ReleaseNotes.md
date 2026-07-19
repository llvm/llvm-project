<!-- This document is written in Markdown and uses extra directives provided by
MyST (https://myst-parser.readthedocs.io/en/latest/). -->

<!-- If you want to modify sections/contents permanently, you should modify both
ReleaseNotes.md and ReleaseNotesTemplate.txt. -->

# LLVM {{env.config.release}} Release Notes

```{contents}
```

````{only} PreRelease
```{warning} These are in-progress notes for the upcoming LLVM {{env.config.release}}
             release. Release notes for previous releases can be found on
             [the Download Page](https://releases.llvm.org/download.html).
```
````

## Introduction

This document contains the release notes for the LLVM Compiler Infrastructure,
release {{env.config.release}}.  Here we describe the status of LLVM, including
major improvements from the previous release, improvements in various subprojects
of LLVM, and some of the current users of the code.  All LLVM releases may be
downloaded from the [LLVM releases web site](https://llvm.org/releases/).

For more information about LLVM, including information about the latest
release, please check out the [main LLVM web site](https://llvm.org/).  If you
have questions or comments, the [Discourse forums](https://discourse.llvm.org)
is a good place to ask them.

Note that if you are reading this file from a Git checkout or the main
LLVM web page, this document applies to the *next* release, not the current
one.  To see the release notes for a specific release, please see the
[releases page](https://llvm.org/releases/).

## Non-comprehensive list of changes in this release

<!-- For small 1-3 sentence descriptions, just add an entry at the end of
this list. If your description won't fit comfortably in one bullet
point (e.g. maybe you would like to give an example of the
functionality, or simply have a lot to talk about), see the comment below
for adding a new subsection. -->

* ...

<!-- If you would like to document a larger change, then you can add a
subsection about it right here. You can copy the following boilerplate:

### Special New Feature

Makes programs 10x faster by doing Special New Thing.
-->

### Changes to the LLVM IR

### Changes to LLVM infrastructure

### Changes to building LLVM

### Changes to TableGen

* `!cond` operator short-circuits at the first `true` condition.  Subsequent
  `condition : value` pairs, along with their corresponding side effects,
  are left unresolved.

### Changes to Interprocedural Optimizations

### Changes to Vectorizers

### Changes to the AArch64 Backend

* On AArch64 Windows targets, return address signing now uses the B-key by
  default because Windows unwind information only supports B-key signing.

### Changes to the AMDGPU Backend

### Changes to the ARM Backend

### Changes to the AVR Backend

### Changes to the DirectX Backend

### Changes to the Hexagon Backend

### Changes to the LoongArch Backend

### Changes to the MIPS Backend

### Changes to the PowerPC Backend

### Changes to the RISC-V Backend

### Changes to the WebAssembly Backend

### Changes to the Windows Target

### Changes to the X86 Backend

### Changes to the OCaml bindings

### Changes to the Python bindings

### Changes to the C API

### Changes to the CodeGen infrastructure

### Changes to the Metadata Info

### Changes to the Debug Info

### Changes to the LLVM tools

### Changes to LLDB

#### Windows

* Python 3.11 or later is now recommended for building LLDB 23 on Windows. From LLDB 24, Python 3.11 or later will be required.
* Messages from `OutputDebugString[A|W]` are now shown inline when using LLDB
  from the command-line and in the output window when using lldb-dap.
* LLDB now uses `lldb-server.exe` to launch and manage the program being debugged,
  instead of running it within LLDB's own process. To revert to the previous behavior, set the environment variable `LLDB_USE_LLDB_SERVER=0`.
* Support for PDB symbol servers has been added. By default, no symbol servers are used.
  You can control this either through the [`_NT_SYMBOL_PATH`](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/symbol-path)
  environment variable or by setting `plugin.symbol-locator.symstore.urls`
  (see [`plugin.symbol-locator.symstore`](https://lldb.llvm.org/use/settings.html#symstore) for more info).
* LLDB no longer depends on the Python private API on Windows. Users are now free to
  use any Python version they want, as long as it is 3.8 or later and LLDB can find it
  (i.e. it is on their `PATH`).

### Changes to BOLT

### Changes to Sanitizers

### Other Changes

## External Open Source Projects Using LLVM {{env.config.release}}

## Additional Information

A wide variety of additional information is available on the
[LLVM web page](https://llvm.org/), in particular in the
[documentation](https://llvm.org/docs/) section.  The web page also contains
versions of the API documentation which is up-to-date with the Git version of
the source code.  You can access versions of these documents specific to this
release by going into the `llvm/docs/` directory in the LLVM tree.

If you have any questions or comments about LLVM, please feel free to contact
us via the [Discourse forums](https://discourse.llvm.org).
