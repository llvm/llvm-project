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

Emits only the **final** remark for each location. This is useful in multi-pass
compilers where an early pass may report a failure, but a later pass succeeds.

**Example:** Only the successful remark is emitted:

```c++
auto opts = remark::RemarkOpts::name("Unroller").category("LoopUnroll");

// First pass: reports failure
remark::failed(loc, opts) << "Loop could not be unrolled";

// Later pass: reports success (this is the one emitted)
remark::passed(loc, opts) << "Loop unrolled successfully";
```

You can also implement custom policies by inheriting from the policy interface.

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
