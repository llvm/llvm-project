# Remark Infrastructure

Remarks are **structured, human- and machine-readable notes** emitted by the
compiler to communicate:

- What transformations were applied
- What optimizations were missed
- Why certain decisions were made

The **`RemarkEngine`** collects remarks during compilation and routes them to a
pluggable **streamer**. By default, MLIR integrates with LLVM's
[`llvm::remarks`](https://llvm.org/docs/Remarks.html) infrastructure, enabling
you to:

- Stream remarks as passes run
- Serialize to **YAML** or **LLVM Bitstream** 

***

## Overview

- **Opt-in** – Disabled by default; zero overhead unless enabled.
- **Per-context** – Configured on `MLIRContext`.
- **Formats** – LLVM Remark engine (YAML / Bitstream) or custom streamers.
- **Kinds** – `Passed`, `Missed`, `Failure`, `Analysis`.
- **API** – Lightweight streaming interface using `<<` (like MLIR diagnostics).

***

## Architecture

The remark system consists of two main components:

### RemarkEngine

Owned by `MLIRContext`, the engine:

- Receives finalized `InFlightRemark` objects
- Optionally mirrors remarks to the `DiagnosticEngine`
- Dispatches to the installed streamer

### MLIRRemarkStreamerBase

An abstract backend interface with a single hook:

```c++
virtual void streamOptimizationRemark(const Remark &remark) = 0;
```

The default implementation, **`MLIRLLVMRemarkStreamer`**, adapts `mlir::Remark`
to LLVM's remark format and writes YAML or Bitstream via
`llvm::remarks::RemarkStreamer`.

**Ownership chain:** `MLIRContext` → `RemarkEngine` → `MLIRRemarkStreamerBase`

***

## Remark Categories

MLIR provides four built-in categories:

### Passed

An optimization or transformation succeeded.

```
[Passed] RemarkName | Category:Vectorizer:myPass1 | Function=foo | Remark="vectorized loop", tripCount=128
```

### Missed

An optimization didn't apply and produces ideally an actionable feedback.

```
[Missed]  | Category:Unroll | Function=foo | Reason="tripCount=4 < threshold=256", Suggestion="increase unroll to 128"
```

### Failure

An optimization was attempted but failed. Unlike `Missed`, this indicates an
active attempt that couldn't complete.

For example, when a user requests `--use-max-register=100` but the allocator
cannot satisfy the constraint:

```
[Failed] Category:RegisterAllocator | Reason="Limiting to use-max-register=100 failed; it now uses 104 registers for better performance"
```

### Analysis

Neutral informational output—useful for profiling and debugging.

```
[Analysis] Category:Register | Remark="Kernel uses 168 registers"
[Analysis] Category:Register | Remark="Kernel uses 10kB local memory"
```

***

## Emitting Remarks

Use the `remark::*` helpers to create an **in-flight remark**, then append
content with the `<<` operator.

### Configuring Remark Options

Each remark accepts four fields (all `StringRef`):

| Field          | Description                                    |
|***************-|************************************************|
| **Name**       | Identifiable name for the remark               |
| **Category**   | High-level classification                      |
| **Sub-category** | Fine-grained classification                  |
| **Function**   | The function where the remark originates       |

### Basic Example

```c++
#include "mlir/IR/Remarks.h"

LogicalResult MyPass::runOnOperation() {
  Location loc = getOperation()->getLoc();

  auto opts = remark::RemarkOpts::name("VectorizeLoop")
                  .category("Vectorizer")
                  .subCategory("MyPass")
                  .function("foo");

  // Passed: transformation succeeded
  remark::passed(loc, opts)
      << "vectorized loop"
      << remark::metric("tripCount", 128);

  // Analysis: informational output
  remark::analysis(loc, opts)
      << "Kernel uses 168 registers";

  // Missed: optimization skipped (with reason and suggestion)
  remark::missed(loc, opts)
      << remark::reason("tripCount={0} < threshold={1}", 4, 256)
      << remark::suggest("increase unroll factor to {0}", 128);

  // Failure: optimization attempted but failed
  remark::failed(loc, opts)
      << remark::reason("unsupported pattern encountered");

  return success();
}
```

***

## Metrics and Helpers

All helper functions accept
[LLVM format strings](https://llvm.org/docs/ProgrammersManual.html#formatting-strings-the-formatv-function),
which build lazily—ensuring zero cost when remarks are disabled.

| Helper                         | Description                              |
|******************************--|******************************************|
| `remark::metric(key, value)`   | Adds a structured key–value pair         |
| `remark::add(fmt, ...)`        | Shortcut for `metric("Remark", ...)`     |
| `remark::reason(fmt, ...)`     | Shortcut for `metric("Reason", ...)`     |
| `remark::suggest(fmt, ...)`    | Shortcut for `metric("Suggestion", ...)` |

### String Shorthand

Appending a plain string:

```c++
remark::passed(loc, opts) << "vectorized loop";
```

is equivalent to:

```c++
remark::passed(loc, opts) << remark::metric("Remark", "vectorized loop");
```

### Custom Metrics

Add structured data for machine readability:

```c++
remark::passed(loc, opts)
    << "loop optimized"
    << remark::metric("TripCount", 128)
    << remark::metric("VectorWidth", 4);
```

***

## Emitting Policies

The `RemarkEngine` supports pluggable policies that control which remarks are
emitted.

### RemarkEmittingPolicyAll

Emits **all** remarks unconditionally.

### RemarkEmittingPolicyFinal

Stores remarks until `finalize()` is called and keeps the last report for each
identity: the complete location, remark name, combined category and subcategory,
and remark kind. The function name is not part of the identity. Reports with
different kinds, such as a missed remark followed by a passed remark, both
survive.

For example, only the second missed remark is emitted:

```c++
auto opts = remark::RemarkOpts::name("Unroller").category("LoopUnroll");

remark::missed(loc, opts) << "Loop trip count is unknown";

// A later report replaces the earlier remark with the same identity.
remark::missed(loc, opts) << "Loop trip count is too small";
```

`finalize()` drains one batch of stored remarks. Traversals start from roots
with IDs in ID order, then ID-less roots in unspecified order, then any unvisited
remarks in ID order. Related remarks are grouped after their parent.
Shared targets, transitive links, and cycles emit every surviving remark once.
Calling `finalize()` again emits only remarks reported since the previous drain.

If a related target was overwritten in the current batch, its link is removed
from the surviving consumer and its serialized arguments. The consumer still
emits; its link is not redirected to the replacement. Explicit IDs that were not
overwritten in that batch are preserved, including IDs from earlier batches and
IDs unknown to the policy. Callers are responsible for the validity of those
references. The policy releases the discarded-ID bookkeeping after each batch.
An ordinary user argument named `RelatedTo` is preserved independently of typed
remark links.

Reports may run concurrently. Call `finalize()` after reporting stops; it must
not overlap other policy operations, including calls from streamer callbacks.
When concurrent reports have the same identity, the last report to acquire the
policy lock wins. ID order does not guarantee reproducibility across thread
schedules. The built-in LLVM streamer serializes writes; custom streamers used
with `All` must support concurrent calls.

You can also implement custom policies by inheriting from the policy interface.

***

## Querying Enabled Remarks

A remark is only built when its kind and category match the configured
filters. The `remark::passed`, `remark::missed`, `remark::failed` and
`remark::analysis` helpers perform this check themselves, so passes do not need
to do anything. Code that produces remarks from another source, for example a
bridge that imports remarks from a different compiler, can ask the engine what
is enabled before doing any work:

```c++
remark::detail::RemarkEngine *engine = context.getRemarkEngine();
if (!engine || !engine->isAnyRemarkEnabled())
  return; // No remark engine, or no category filter is active.

if (engine->isRemarkEnabled(remark::RemarkKind::RemarkPassed, "Vectorizer"))
  remark::passed(loc, opts) << "vectorized loop";
```

| Query                                        | Answers                                              |
|----------------------------------------------|------------------------------------------------------|
| `isAnyRemarkEnabled()`                       | At least one category filter is active               |
| `isAnyRemarkEnabled(category)`               | Some kind of remark is enabled for the category      |
| `isRemarkEnabled(kind, category)`            | Remarks of `kind` are enabled for the category       |
| `is{Passed,Missed,Analysis,Failed}OptRemarkEnabled(category)` | Per-kind query                      |

***

## Searching and Linking Remarks

History is opt-in. When enabled, the engine records the ID, name, category,
subcategory, function, kind, and location of every emitted remark, and
`findRemarks(query)` returns the matching IDs in ID order. Under `Final`,
pending candidates are searchable as well. They may be replaced before emission;
once replaced, they no longer match.

`RemarkQuery` matches exact values: `category()` the base category, `at()` the
complete location, wrappers included. Unset fields match anything; an empty
string matches an empty field.

```c++
auto *engine = context.getRemarkEngine();
if (!engine)
  return;
engine->enableHistory();
auto analysisOpts = remark::RemarkOpts::name("VectorSizeInference")
                        .category("Vectorizer")
                        .function("kernel");
remark::analysis(loc, analysisOpts)
    << remark::reason("Unable to infer vector sizes");

auto query = remark::RemarkQuery::named("VectorSizeInference")
                 .category("Vectorizer")
                 .function("kernel")
                 .kind(remark::RemarkKind::RemarkAnalysis)
                 .at(loc);
remark::missed(loc, remark::RemarkOpts::name("Vectorization")
                       .category("Vectorizer")
                       .function("kernel"))
    << remark::relatedTo(query);
```

`relatedTo(query)` links the remark to every match at the time of the `<<`,
each once. With no match, or with history disabled, it adds nothing. Under
`Final`, a link whose target is replaced in the same finalization batch is
dropped, also when the replacement comes from another function or is the linking
remark itself. A match indicates a relationship, not causality, and the IR may
have changed since.

Each query scans all pending and emitted remarks, and a broad query in a loop
produces many links; a known ID links without a query through
`RemarkOpts::relatedTo(id)`. Each link is serialized as its own `RelatedTo`
argument.

Reports and queries may run concurrently. Under `Final`, they must finish before
`finalize()` starts and may resume after it returns; its streamer callbacks must
not report or query remarks. Under `All`, streamers run on the reporting threads.
With parallel passes, cross-function results depend on timing; scope queries
with `function()` or `at()`.

`clearHistory()` discards the recorded metadata; `enableHistory(false)` also
stops recording and makes queries return nothing. Neither touches pending
remarks or existing links.

Pending IDs describe the current candidates. If a consumer is reported only
after its target's batch has drained, Final no longer knows whether that target
was replaced. Callers carrying IDs across batches must ensure those targets
were emitted.

***

## Enabling Remarks

### Option 1: LLVM Remark Streamer (YAML or Bitstream)

Persist remarks to a file for post-processing:

```c++
// Setup categories
remark::RemarkCategories cats{
    /*passed=*/   "LoopUnroll",
    /*missed=*/   std::nullopt,
    /*analysis=*/ std::nullopt,
    /*failed=*/   "LoopUnroll"
};

// Use final policy
std::unique_ptr<remark::RemarkEmittingPolicyFinal> policy =
        std::make_unique<remark::RemarkEmittingPolicyFinal>();

remark::enableOptimizationRemarksWithLLVMStreamer(
    context, outputFile, llvm::remarks::Format::YAML, std::move(policy), cats);
```

**YAML output** (human-readable):

```yaml
*** !Passed
pass:     Vectorizer:MyPass
name:     VectorizeLoop
function: foo
loc:      input.mlir:12:3
args:
  - Remark:    vectorized loop
  - tripCount: 128
```

**Bitstream format** — compact binary for large-scale analysis.

### Option 2: Diagnostic Engine (No Streamer)

Mirror remarks to the standard diagnostic output:

```c++
// Setup categories
remark::RemarkCategories cats{
    /*passed=*/   "LoopUnroll",
    /*missed=*/   std::nullopt,
    /*analysis=*/ std::nullopt,
    /*failed=*/   "LoopUnroll"
};

// Use final policy
std::unique_ptr<remark::RemarkEmittingPolicyFinal> policy =
        std::make_unique<remark::RemarkEmittingPolicyFinal>();

remark::enableOptimizationRemarks(
    context,
    /*streamer=*/ nullptr,
    /*policy=*/ std::move(policy),
    cats,
    /*printAsEmitRemarks=*/ true);
```

### Option 3: Custom Streamer

Implement your own backend for specialized output formats:

```c++
class MyStreamer : public MLIRRemarkStreamerBase {
public:
  void streamOptimizationRemark(const Remark &remark) override {
    // Custom serialization logic
  }
};

auto streamer = std::make_unique<MyStreamer>();
remark::enableOptimizationRemarks(context, std::move(streamer), cats);
```

***

## Importing LLVM Remarks

LLVM passes emit their own
[optimization remarks](https://llvm.org/docs/Remarks.html) while MLIR compiles
a module with LLVM, for example in `gpu-module-to-binary`. `ModuleToObject`
installs a `remark::LLVMToMLIRDiagnosticHandler` on the `llvm::LLVMContext` it
compiles with, which imports these remarks into the remark engine: the category
is the LLVM pass name prefixed with `llvm-`, the LLVM remark name becomes the
name, the message is stored under the `Remark` argument and the structured LLVM
arguments are copied as metrics. The remark filters therefore select LLVM
remarks by pass name, and `llvm-.*` selects all of them:

```
mlir-opt kernel.mlir --gpu-module-to-binary="format=isa" --remarks-filter-passed=llvm-inline
```

```
[Passed] Inlined | Category:llvm-inline | Function=kernel | Callee=helper, Caller=kernel, Cost=-30, Remark="'helper' inlined into 'kernel' with (cost=-30, threshold=337)", Threshold=337
```

Remarks that are not enabled in the engine are left to LLVM, so LLVM's
`-pass-remarks` flags keep working. Other LLVM diagnostics, such as errors
reported by a backend, become MLIR diagnostics.

LLVM remarks carry a source location only when the LLVM IR has debug
information; run `ensure-debug-info-scope-on-llvm-func` before serializing to
get `file:line:column` locations. Otherwise a remark is attached to the MLIR
symbol named after its LLVM function, or to the serialized operation.

Remarks saved by an LLVM-based compiler to a YAML or bitstream file can be
replayed into the engine with `remark::importLLVMRemarks`, which applies the
same conversion.
