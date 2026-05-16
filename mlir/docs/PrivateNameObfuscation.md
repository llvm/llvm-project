# Private Name Obfuscation

[TOC]

This page documents an opt-in TableGen + CMake mechanism for replacing the
human-readable names of dialects, operations, attributes, types, and passes
in a built MLIR binary with opaque identifiers. The intent is to make it
harder to reverse-engineer a release binary while preserving a fully-readable
internal/test build from the same source tree.

The whole obfuscation logic is delegated to an external shell command
supplied by the build, so MLIR itself does not embed any particular hash
function or secret material.

## What gets obfuscated

Three knobs control what gets obfuscated:

* `-DMLIR_PRIVATE_NAME_OBFUSCATOR=<shell-cmd>` — the external command used
  to obfuscate one name at a time.
* `-DMLIR_PRIVATE_DIALECTS=<d1;d2;...>` — the (semicolon-separated) list of
  dialect namespaces whose items should be obfuscated.
* `-DMLIR_PRIVATE_PASSES=ON` — when set, treat *every* pass as private.

Whether a given dialect or pass is "private" is a property of the tool
consuming it, not of the source definition, so both knobs are supplied by
the build rather than authored in ODS.

When `MLIR_PRIVATE_NAME_OBFUSCATOR` is set together with either of the
other two, the matching items get the literals below replaced with opaque
identifiers:

| Source                     | Affected literal(s)                                          |
| -------------------------- | ------------------------------------------------------------ |
| `Dialect`                  | `getDialectNamespace()`, dialect constructor                 |
| `Op`                       | `getOperationName()`, adaptor `odsOpName`, error prefixes |
| `AttrDef`, `TypeDef`       | `getMnemonic()`, `name`, `dialectName`, alias `getAlias`     |
| `PassBase` / `Pass`        | `getArgument()`, `getArgumentName()`, `getName()`, `getPassName()` |

For each unique name `N`, mlir-tblgen runs roughly:

```sh
printf '%s' 'N' | <shell-cmd>
```

It then takes the first whitespace-delimited token from stdout, prefixes
it with `_` (so the result is always a valid MLIR mnemonic / C++ identifier),
and uses that as the obfuscated name. Results are cached, so the command is
invoked at most once per unique name within a single mlir-tblgen invocation.

For op / attribute / type names, the obfuscator is invoked separately on
the dialect prefix and on the mnemonic, and the results are rejoined with
a dot, so the runtime helper `OperationName::getDialectNamespace()` (which
splits at the first `.`) keeps working.

Obfuscated names are **deterministic** as long as the configured command
is deterministic, so all translation units within one build agree on the
obfuscated spelling. Pattern matching, `ConversionTarget`, the bytecode
reader/writer (which uses whatever `getDialectNamespace()` /
`getOperationName()` return), and dialect registration all keep working
without source changes.

Private ops also behave as if no `assemblyFormat` or
`hasCustomAssemblyFormat` was specified when private-name obfuscation is
enabled. ODS does not generate the custom/declarative `parse` and `print`
methods for those ops. They still print in generic form, using the
obfuscated operation name, and their custom textual syntax is rejected.

Op argument names (the keys returned by `getAttributeNames()` / used by
generated `StringAttr` accessors) are **not** obfuscated. They are dictionary
keys with established meaning across patterns, properties, and serialization,
so they remain in their original form even on private ops. Raw string lookups
such as `op->getAttr("foo")` therefore keep working unchanged.

When `MLIR_PRIVATE_PASSES=ON`, the descriptions returned by
`getDescription()`, the per-option `cl::desc(...)` text, and the
per-statistic description text are also emitted as empty strings (since
plaintext descriptions would otherwise leak the meaning of the obfuscated
name). Per-pass `register{PassName}()` helpers and the
`mlirCreate{Group}{PassName}` / `mlirRegister{Group}{PassName}` C-API
entries keep being emitted under their original C++ class names; they
register and invoke the pass under its obfuscated textual argument and
display name.

## Build configuration

```cmake
# Public/test/dev build: no obfuscation (default).
cmake -G Ninja path/to/llvm \
  -DLLVM_ENABLE_PROJECTS=mlir

# Release build: obfuscate the listed dialects and all passes using an
# external obfuscator command.
cmake -G Ninja path/to/llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DMLIR_PRIVATE_NAME_OBFUSCATOR="md5sum | awk '{print \$1}'" \
  -DMLIR_PRIVATE_DIALECTS="my_internal_dialect;my_lowering_dialect" \
  -DMLIR_PRIVATE_PASSES=ON
```

`MLIR_PRIVATE_NAME_OBFUSCATOR` controls private-name obfuscation: leave it
empty to disable obfuscation, or set it to a shell command. The command
receives each name on stdin (no trailing newline) and should write the
obfuscated form to stdout. mlir-tblgen uses the first whitespace-delimited
token of that output as the obfuscated name, so simple tools that emit
extra fields (such as `md5sum`, which prints `<hash>  -`) work out of the
box.

Choose any deterministic, build-stable obfuscator for production builds.
The obfuscator must produce a unique output per unique input (no
collisions); hashing tools usually satisfy this in practice but it is the
user's responsibility to pick something appropriate. Examples:

* `md5sum`
* `sha256sum`
* An HMAC pipeline with a build-private key:
  `openssl dgst -sha256 -hmac "$RELEASE_SALT" -hex | awk '{print $2}'`
* A custom script that mixes in a vendor-specific salt before hashing.
* `mlir/utils/private-name-obfuscator-example.py` — a minimal Python
  script bundled with MLIR. It assigns each unique input the next free
  positive integer and persists the mapping in a file (one name per
  line; line N is the original name whose obfuscated form is `_N`). The
  same file then doubles as the de-obfuscation table for translating
  customer-reported `_42`-style identifiers back to human-readable
  names. Suitable as a starting point for downstream tooling.

The obfuscator command and the dialect list are fed directly to the
`mlir-tblgen` command line as `--mlir-private-name-obfuscator=<cmd>` and
`--mlir-private-dialects=<csv>`. They are **not** embedded in the resulting
binary. Bytecode produced by a build with one (obfuscator, dialect-list)
pair is generally unreadable by a build that uses a different pair, so
pin both across releases that need binary compatibility.

## Marking items as private

Neither dialects nor passes are marked private in ODS — whether to obfuscate
them depends on the tool consuming them (e.g., `scf` and `Canonicalizer` may
be private in one downstream compiler and public in another). Privacy is
configured entirely by the build:

* A dialect becomes private by being listed in `MLIR_PRIVATE_DIALECTS`. All
  ops, attributes, and types declared by a private dialect are obfuscated;
  there is no per-op or per-def override.
* All passes become private when `MLIR_PRIVATE_PASSES=ON`. The toggle is
  global — there is no per-pass override either.

```tablegen
def MyDialect : Dialect {
  let name = "mydialect";
  let cppNamespace = "::my";
}

def MyOp : Op<MyDialect, "do_thing", []>;   // obfuscated iff MyDialect is
                                            // in MLIR_PRIVATE_DIALECTS

def MyPass : Pass<"my-pass"> {              // obfuscated iff
  let summary = "Does the thing.";          // MLIR_PRIVATE_PASSES=ON
}
```

## Caveats

* Hand-written code that compares `op->getName().getStringRef()` against a
  spelled-out op name (e.g. `== "mydialect.do_thing"`) breaks under
  obfuscation. Migrate such checks to `isa<my::DoThingOp>()` (which uses
  TypeID and is unaffected) or compare against
  `my::DoThingOp::getOperationName()` (which is itself obfuscated, so the
  comparison still works).
* Diagnostics and verifier messages naturally print the obfuscated names
  because they use runtime `getName()` / `getDialect()->getNamespace()`
  calls. The English skeleton text around the names is not stripped.
* Python bindings emitted by `gen-python-op-bindings` are not adjusted by
  this mechanism. Do not generate Python bindings for private dialects or
  ops.
* `LLVM_DEBUG`, `LDBG`, statistics (`LLVM_ENABLE_STATS`), and `--debug`
  paths are already removed from a release build (`NDEBUG`); they don't
  need separate handling.
* The obfuscator is spawned via `popen` (or `_popen` on Windows), so the
  command must be runnable by the host's default shell, and must be on
  `PATH` when mlir-tblgen runs.

## Audit checklist for downstream trees

Before enabling private-name obfuscation in a downstream compiler, audit for
hand-written string comparisons and textual pipeline dependencies. These
patterns should generally be rewritten to use TypeID-based APIs, concrete op
classes, or the generated `::getOperationName()` accessors:

```sh
rg 'getName\(\)\.getStringRef\(\).*==|==.*getName\(\)\.getStringRef\(\)' path/to/downstream
rg 'OperationName\("[^"]+"' path/to/downstream
rg 'RegisteredOperationName::lookup\("[^"]+"' path/to/downstream
rg 'getOrLoadDialect\("[^"]+"' path/to/downstream
rg 'PassInfo::lookup\("[^"]+"' path/to/downstream
rg 'parsePassPipeline|--pass-pipeline|register.*Passes' path/to/downstream
```

Comparisons against generated names, such as
`my::DoThingOp::getOperationName()`, remain valid because the generated
method returns the obfuscated spelling in release builds.
