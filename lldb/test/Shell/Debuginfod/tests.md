# Tests for basic Debuginfod functionality

Because the Debuginfod protocol is a simple HTTP path-based system, one can
mimic a Debuginfod server by setting up a directory structure to reflect the
protocol properly. That's how all these tests operate. We override the default
`DEBUGINFOD_URLS` property with a `file://` URL and populate it with the symbol
files we need for testing.

## Baseline capabilities

Things to test:

- Just for sanity, the `-no-locator` tests validate that lldb works as the test
  expects if the files that Debuginfod _should_ provide are already there.
- Validate that symbols _aren't_ found without Debuginfod, to ensure they
  weren't cached from previous runs (`-negative` tests).
- Validate that the Debuginfod symbol locator is providing the expected symbols
  when the debugger doesn't already have them available.

## Symbol "Configurations" to test

There are 2 top level configurations, each of which then get some post-processing.
- The 'normal' build: a simple binary built with debug information (`-g`)
- The 'split' build: A binary built with `-gsplit-dwarf` that produces `.dwo` and `.dwp` files

For the normal build there are 2 post-processing steps where Debuginfod may be used.
- Strip the binary while hosting the full, unstripped binary on a Debuginfod server.
- Strip the binary and produce a corresponding 'only-keep-debug' version of the binary to host on a Debuginfod server.

For the split build, there are *3* post-processing outcomes where Debuginfod usage makes sense:
- Stripped binary, hosting the full unstripped binary on a Debuginfod server, along with it's `.dwp` file.
- Stripped binary, hosting the 'only-keep-debug' version of the binary *and* it's `.dwp` file.
- Unstripped binary, hosting only the `.dwp` file on the Debuginfod server.

For both normal and split builds, when using an 'only-keep-debug' symbol file, the original file
also requires the addition of the `.gnu.debuglink` section (which can be added using `objcopy`)

### Troubles

The Debuginfod protocol is somewhat underspecified regarding what the 2 different callbacks actually return. The protocol itself is quite simple: query an HTTP server with the path `buildid/{.note.gnu.build-id hash}/debuginfo` and `buildid/{.note.gnu.build-id hash}/executable`. A keen reader may note that there's one configuration (split, only-keep-debug) where you need two different files to properly symbolicate, but the API only supports a single debuginfo file. For that scenario, I've chosen to simply treat the `.dwp` file as "debuginfo" and the "only-keep-debug" stripped binary as "executable". This scenario doesn't actually work at all in GDB. It seems more straightforward than trying to extend the protocol.

### Stripped:

- Should ask for symbols from the service; can be registered as 'executable' and
  they should work!

### Stripped "only-keep-debug":

- Should get them from the debuginfo, not from the executable query

### split-dwarf:

- Should get dwp file from debuginfo

### split-dwarf + stripped:

- Should get dwp from debuginfo, and the unstripped binary from executable


image
lldb.target.ResolveLoadAddress
