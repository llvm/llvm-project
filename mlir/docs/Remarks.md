# Remark Infrastructure

[TOC]

Optimization remarks are structured, machine- and human-readable notes emitted by passes to explain what was optimized, what was missed, and why. MLIR integrates LLVM’s **remarks** infrastructure to make these insights easy to produce and consume.

**Key points**

- **Opt-in**: Disabled by default. No cost unless enabled.
- **Per-context**: Configured on `MLIRContext`.
- **Formats**: YAML or LLVM remark bitstream.
- **Kinds**: Pass, Missed, Failure, Analysis.
- **API**: Lightweight stream interface (similar to diagnostics) with `<<`.

## Enabling remarks

Enable once per `MLIRContext` (e.g., in your tool, pass pipeline setup, or test):

```c++
#include "mlir/IR/MLIRContext.h"

// Writes remarks to /tmp/remarks.yaml in YAML format and mirrors them to
// the DiagnosticEngine as 'remark' diagnostics with the given category labels.
context.setupOptimizationRemarks(
    /*outputPath=*/"/tmp/remarks.yaml",
    /*outputFormat=*/yaml,              // or "bitstream"
    /*printAsEmitRemarks=*/true,
    /*categoryPassName=*/"opt.pass",      // optional category labels for mirroring
    /*categoryMissName=*/"opt.missed",
    /*categoryAnalysisName=*/"opt.analysis",
    /*categoryFailedName=*/"opt.failed");
```

### Emitting remarks from a pass

The functions `reportOptimization*` return an in-flight remark object (like MLIR diagnostics). One can append strings and key–value pairs with <<. 

```c++
#include "mlir/IR/Remarks.h"

using namespace mlir;

LogicalResult MyPass::runOnOperation() {
  Operation *op = getOperation();
  Location loc = op->getLoc();

  // PASS: something succeeded
  reportOptimizationPass(loc, /*category=*/"vectorizer", /*passName=*/"MyPass")
      << "vectorized loop with tripCount="
      << RemarkBase::RemarkKeyValue("tripCount", 128);

  // ANALYSIS: neutral insight
  reportOptimizationAnalysis(loc, "unroll", "MyPass")
      << "estimated cost: " << RemarkBase::RemarkKeyValue("cost", 42);

  // MISSED: explain why + suggest a fix
  reportOptimizationMiss(loc, "unroll", "MyPass",
                         /*suggestion=*/"increase unroll factor to >=4")
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

## Output formats

#### YAML

A typical remark serialized to YAML looks like following. It is Readable, easy to diff and grep.

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

Compact binary format supported by LLVM’s remark tooling. Prefer this for large production runs or where existing infrastructure already understands LLVM remarks.
