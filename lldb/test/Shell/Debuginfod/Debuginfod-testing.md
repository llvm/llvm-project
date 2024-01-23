# Tests for basic Debuginfod functionality

Because the Debuginfod protocol is a simple HTTP path-based system, one can
mimic a Debuginfod server by setting up a directory structure to reflect the
protocol properly. That's how all these tests operate. We override the default
`DEBUGINFOD_URLS` property with a `file://` URL and populate it with the symbol
files we need for testing.

## What's being tested

- For assumption validation, the `*-no-locator` tests verify that lldb works as
  the test expects when files that Debuginfod should provide (`.dwp` files,
  `.gnu.debuglink`'ed files, etc...) are _already_ there.
- The `*-negative` tests validate that symbols _aren't_ found without
  Debuginfod, to ensure they haven't been cached from previous runs (in the
  hopes of preventing false positive testing).
- The `*-positive*` tests check that the Debuginfod symbol locator is providing
  the expected symbols when the debugger doesn't already have them available.

### Symbol file variations tested

There are 5 variations of symbol data where Debuginfod provides value:

1. The `strip` build variation is a binary built with debug information (`-g`),
   but stripped for deployment. The Debuginfod service can then host the
   unstripped binary (as either `executable` or `debuginfo`).
2. The `okdstrip` build variation is a binary build with `-g`, stripped for
   deployment, where the Debuginfod service is hosting the output of
   `objcopy --only-keep-debug` (which should also be linked to the stripped file
   using `--add-gnu-debuglink`). Again, the file could be hosted as either
   `executable` or `debuginfo`.
3. The `split` build variation is a binary built with `-gsplit-dwarf` that
   produces `.dwo` which are subsequently linked together (using `llvm-dwp`)
   into a single `.dwp` file. The Debuginfod service hosts the `.dwp` file as
   `debuginfo`.
4. The `split-strip` build variation is a binary built with `-gsplit-dwarf`,
   then stripped in the same manner as variation #1. For this variation,
   Debuginfod hosts the unstripped binary as `executable` and the `.dwp` file as
   `debuginfo`.
5. The `split-okdstrip` build variation is the combination of variations 2 and
   3, where Debuginfod hosts the `.gnu.debuglink`'ed file as `executable` and
   the `.dwp` as `debuginfo`.

### Lack of clarity/messy capabilities from Debuginfod

The [debuginfod protocol](https://sourceware.org/elfutils/Debuginfod.html) is
underspecified for some variations of symbol file deployment. The protocol
itself is quite simple: query an HTTP server with the path
`buildid/{.note.gnu.build-id hash}/debuginfo` or
`buildid/{.note.gnu.build-id hash}/executable` to acquire "symbol data" or "the
executable". Where there is lack of clarity, I prefer requesting `debuginfo`
first, then falling back to `executable` (Scenarios #1 & #2). For Scenario #5,
I've chosen to expect the stripped (i.e. not full) executable, which contains a
number of sections necessary to correctly symbolicate will be hosted from the
`executable` API. Depending upon how Debuginfod hosting services choose to
support `.dwp` paired with stripped files, these assumptions may need to be
revisited.

I've also chosen to simply treat the `.dwp` file as `debuginfo` and the
"only-keep-debug" stripped binary as `executable`. This scenario doesn't appear
to work at all in GDB. Supporting it how I did seems more straight forward than
trying to extend the protocol. The protocol _does_ support querying for section
contents by name for a given build ID, but adding support for that in LLDB
looks...well beyond my current capability (and LLVM's Debuginfod library doesn't
support it at this writing, anyway).
