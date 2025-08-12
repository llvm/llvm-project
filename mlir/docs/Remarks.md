# Remark Infrastructure

[TOC]

Remarks are structured, human- and machine-readable notes emitted by compiler to
explain what was transformed, what was missed, and why. The `RemarkEngine`
collects finalized remarks during compilation and forwards them to a pluggable
streamer. A default streamer integrates LLVM’s `llvm::remarks` so you can stream
while a pass runs and serialize to disk (YAML or LLVM bitstream) for tooling.

**Key points**

- **Opt-in**: Disabled by default; zero overhead unless enabled.
- **Per-context**: Configured on `MLIRContext`.
- **Formats**: Custom streamers, or LLVM’s Remark engine (YAML / Bitstream).
- **Kinds**: `Passed`, `Missed`, `Failure`, `Analysis`.
- **API**: Lightweight streaming interface with `<<` (similar to diagnostics).

## How it works

Remarks has two important classes:

- **`RemarkEngine`** (owned by `MLIRContext`): receives finalized
  `InFlightRemark`s, optionally mirrors them to the `DiagnosticEngine`, then
  dispatches to the installed streamer.
- **`MLIRRemarkStreamerBase`** (abstract): backend interface with a single hook
  `streamOptimizationRemark(const Remark &)`.

**Default backend – `MLIRLLVMRemarkStreamer`** Adapts `mlir::Remark` to
`llvm::remarks::Remark` and writes YAML/bitstream via
`llvm::remarks::RemarkStreamer` to a `ToolOutputFile`.

**Ownership**: `MLIRContext` → `RemarkEngine` → `MLIRRemarkStreamerBase`.

## Enable Remarks via mlir::emitRemarks (No Streamer)

Enable once per `MLIRContext` (e.g., where you build your pass pipeline or in
your tool). If `printAsEmitRemarks` is true, each remark is also mirrored to the
context’s `DiagnosticEngine` under the provided category labels—handy for
interactive tools and tests.

```c++
mlir::MLIRContext::RemarkCategories cats{/*passed=*/categoryLoopunroll,
                                          /*missed=*/std::nullopt,
                                          /*analysis=*/std::nullopt,
                                          /*failed=*/categoryLoopunroll};

context.enableOptimizationRemarks(/*streamer=*/nullptr,
                                  cats,
                                  /*printAsEmitRemarks=*/true);
```

## Enable Remarks with LLVMRemarkStreamer (YAML/Bitstream)

If you want to persist remarks to a file in YAML or bitstream format, use
`mlir::remark::LLVMRemarkStreamer` (helper shown below):

You can read more information about [LLVM's Remark from here](https://llvm.org/docs/Remarks.html).

```c++
#include "mlir/Remark/RemarkStreamer.h"

mlir::MLIRContext::RemarkCategories cats{/*passed=*/categoryLoopunroll,
                                         /*missed=*/std::nullopt,
                                         /*analysis=*/std::nullopt,
                                         /*failed=*/categoryLoopunroll};

mlir::remark::enableOptimizationRemarksToFile(
    context, yamlFile, llvm::remarks::Format::YAML, cats);
```

## Emitting remarks from a pass

The `reportOptimization*` functions return an in-flight remark object (like MLIR
diagnostics). Append strings and key–value pairs with `<<`.

```c++
#include "mlir/IR/Remarks.h"

using namespace mlir;

LogicalResult MyPass::runOnOperation() {
  Operation *op = getOperation();
  Location loc = op->getLoc();

  // PASS: something succeeded
  reportRemarkPassed(loc, /*category=*/"vectorizer", /*passName=*/"MyPass")
      << "vectorized loop."
      << Remark::RemarkKeyValue("tripCount", 128);

  // ANALYSIS: neutral insight
  reportOptimizationAnalysis(loc, "RegisterCount", "")
      << "Kernel uses 168 registers"

  // MISSED: explain why + suggest a fix
  reportOptimizationMiss(loc, "unroll", "MyPass",
                         /*suggestion=*/[&](){ return "increase unroll factor to >=4"; })
      << "not profitable at this size";

  // FAILURE: action attempted but failed
  if (failed(doThing(op))) {
    reportOptimizationFail(loc, "pipeline", "MyPass")
        << "failed due to unsupported pattern";
    return failure();
  }
  return success();
}
```

### Output formats

#### YAML

Readable, easy to diff and grep.

```yaml
--- !Passed
pass:            MyPass
name:            vectorizer
function:        myFunc
loc:             myfile.mlir:12:3
args:
  - key:         tripCount
    value:       128
message:         "vectorized loop with tripCount=128"
```

#### Bitstream

Compact binary format supported by LLVM’s remark tooling. Prefer this for large
production runs or when existing infrastructure already consumes LLVM remarks.

## Enable Remarks with a Custom Streamer

`RemarkEngine` talks to `MLIRRemarkStreamerBase`. Implement your own streamer to
consume remarks in any format you like:

```c++
class MyStreamer : public MLIRRemarkStreamerBase {
public:
  void streamOptimizationRemark(const Remark &remark) override {
    // Convert Remark to your format and write it out.
  }
};

// ...
auto myStreamer = std::make_unique<MyStreamer>();
context.setupOptimizationRemarks(path,
                                 std::move(myStreamer),
                                 /*printAsEmitRemarks=*/false,
                                 /*categories=*/cat);
```
