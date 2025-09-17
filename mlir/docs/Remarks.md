# Remark Infrastructure

Remarks are **structured, human- and machine-readable notes** emitted by the
compiler to explain:

- What was transformed
- What was missed
- Why it happened

The **`RemarkEngine`** collects finalized remarks during compilation and sends
them to a pluggable **streamer**. By default, MLIR integrates with LLVM’s
[`llvm::remarks`](https://llvm.org/docs/Remarks.html), allowing you to:

- Stream remarks as passes run
- Serialize them to **YAML** or **LLVM bitstream** for tooling

***

## Key Points

- **Opt-in** – Disabled by default; zero overhead unless enabled.
- **Per-context** – Configured on `MLIRContext`.
- **Formats** – LLVM Remark engine (YAML / Bitstream) or custom streamers.
- **Kinds** – `Passed`, `Missed`, `Failure`, `Analysis`.
- **API** – Lightweight streaming interface using `<<` (like MLIR diagnostics).

***

## How It Works

Two main components:

- **`RemarkEngine`** (owned by `MLIRContext`): Receives finalized
  `InFlightRemark`s, optionally mirrors them to the `DiagnosticEngine`, and
  dispatches to the installed streamer.

- **`MLIRRemarkStreamerBase`** (abstract): Backend interface with a single hook:

  ```c++
  virtual void streamOptimizationRemark(const Remark &remark) = 0;
  ```

**Default backend – `MLIRLLVMRemarkStreamer`** Adapts `mlir::Remark` to LLVM’s
remark format and writes YAML/bitstream via `llvm::remarks::RemarkStreamer`.

**Ownership flow:** `MLIRContext` → `RemarkEngine` → `MLIRRemarkStreamerBase`

***

## Categories

MLIR provides four built-in remark categories (extendable if needed):

#### 1. **Passed**

Optimization/transformation succeeded.

```
[Passed] RemarkName | Category:Vectorizer:myPass1 | Function=foo | Remark="vectorized loop", tripCount=128
```

#### 2. **Missed**

Optimization/transformation didn’t apply — ideally with actionable feedback.

```
[Missed]  | Category:Unroll | Function=foo | Reason="tripCount=4 < threshold=256", Suggestion="increase unroll to 128"
```

#### 3. **Failure**

Optimization/transformation attempted but failed. This is slightly different
from the `Missed` category.

For example, the user specifies `-use-max-register=100` when invoking the
compiler, but the attempt fails for some reason:

```bash
$ your-compiler -use-max-register=100 mycode.xyz
```

```
[Failed] Category:RegisterAllocator | Reason="Limiting to use-max-register=100 failed; it now uses 104 registers for better performance"
```

#### 4. **Analysis**

Neutral analysis results.

```
[Analysis] Category:Register | Remark="Kernel uses 168 registers"
[Analysis] Category:Register | Remark="Kernel uses 10kB local memory"
```

***

Here’s a cleaned-up and grammatically improved version of your text:

---

## Emitting Remarks

The `remark::*` helpers return an **`InFlightRemark`**. You can append strings
or key–value metrics using the `<<` operator.

By default, the remark is emitted as soon as the `InFlightRemark` object is
destroyed (typically at the end of its scope). However, it can sometimes be
useful to postpone emitting the remark until the end of the compilation—for example,
to collect all remarks, sort them, and emit them differently later.

If this behavior is desired, the `RemarkEngine` can be used to store postponed remarks.
[See this example of postponing remark emission](#example-postpone-remarks-emitting).

### Remark Options

When constructing a remark, you typically provide four fields that are `StringRef`:

1. **Remark name** – identifiable name
2. **Category** – high-level classification
3. **Sub-category** – more fine-grained classification
4. **Function name** – the function where the remark originates

### Examples

#### Emitting remarks

```c++
#include "mlir/IR/Remarks.h"

LogicalResult MyPass::runOnOperation() {
  Location loc = getOperation()->getLoc();

  remark::RemarkOpts opts = remark::RemarkOpts::name(MyRemarkName1)
                                .category(categoryVectorizer)
                                .function(fName)
                                .subCategory(myPassname1);

  // PASSED
  remark::passed(loc, opts)
      << "vectorized loop"
      << remark::metric("tripCount", 128);

  // ANALYSIS
  remark::analysis(loc, opts)
      << "Kernel uses 168 registers";

  // MISSED (with reason + suggestion)
  int tripBad = 4, threshold = 256, target = 128;
  remark::missed(loc, opts)
      << remark::reason("tripCount={0} < threshold={1}", tripBad, threshold)
      << remark::suggest("increase unroll to {0}", target);

  // FAILURE
  remark::failed(loc, opts)
      << remark::reason("failed due to unsupported pattern");

  return success();
}
```

#### Example: Postpone remarks emitting

`RemarkOpts` has `postpone` option. When it is set, the remark emissing will be
postponed until the end of the compilation.

```c++

  remark::passed(loc, remark::RemarkOpts::name("")
                            .category(categoryLoopunroll)
                            .subCategory(myPassname2)
                            .postpone())
```

***

### Metrics and Shortcuts

Helper functions accept
[LLVM format](https://llvm.org/docs/ProgrammersManual.html#formatting-strings-the-formatv-function)
style strings. This format builds lazily, so remarks are zero-cost when
disabled.

#### Adding Remarks

- **`remark::add(fmt, ...)`** – Shortcut for `metric("Remark", ...)`.

#### Adding Reasons

- **`remark::reason(fmt, ...)`** – Shortcut for `metric("Reason", ...)`. Used to
  explain why a remark was missed or failed.

#### Adding Suggestions

- **`remark::suggest(fmt, ...)`** – Shortcut for `metric("Suggestion", ...)`.
  Used to provide actionable feedback.

#### Adding Custom Metrics

- **`remark::metric(key, value)`** – Adds a structured key–value metric.

Example: tracking `TripCount`. When exported to YAML, it appears under `args`
for machine readability:

```cpp
remark::metric("TripCount", value)
```

#### String Metrics

Passing a plain string (e.g. `<< "vectorized loop"`) is equivalent to:

```cpp
metric("Remark", "vectorized loop")
```

***

## Enabling Remarks with `mlir-opt`

`mlir-opt` provides flags to enable and configure remarks. The available flags
are shown below:

```bash
$> mlir-opt --help
  ...
Remark Options:
Filter remarks by regular expression (llvm::Regex syntax).

  --remark-format=<value>                     - Specify the format for remark output.
    =emitRemark                               -   Print as emitRemark to the command line
    =yaml                                     -   Print as a YAML file
    =bitstream                                -   Print as a bitstream file
  --remarks-filter=<string>                   - Show all remarks: passed, missed, failed, analysis
  --remarks-filter-analyse=<string>           - Show analysis remarks
  --remarks-filter-failed=<string>             - Show failed remarks
  --remarks-filter-missed=<string>             - Show missed remarks
  --remarks-filter-passed=<string>             - Show passed remarks
  --remarks-output-file=<string>               - Output file for YAML and bitstream remark formats (default: mlir-remarks.yaml or mlir-remarks.bitstream)
```

There are five filter flags to select specific categories. The
`--remarks-filter` flag is a general shortcut that applies to all remark
categories. Their usage is as follows:

```bash
# This will only match remarks in the "my-interesting-category" category
mlir-opt --remarks-filter="my-interesting-category"

# You can also use "*" to match categories starting with "my"
mlir-opt --remarks-filter="my*"
```

## Enabling Remarks for downstream compilers via APIs

### 1. **With LLVMRemarkStreamer (YAML or Bitstream)**

Persists remarks to a file in the chosen format.

```c++
mlir::remark::RemarkCategories cats{/*passed=*/categoryLoopunroll,
                                     /*missed=*/std::nullopt,
                                     /*analysis=*/std::nullopt,
                                     /*failed=*/categoryLoopunroll};

mlir::remark::enableOptimizationRemarksWithLLVMStreamer(
    context, yamlFile, llvm::remarks::Format::YAML, cats);
```

**YAML format** – human-readable, easy to diff:

```yaml
--- !Passed
pass:            Category:SubCategory
name:            MyRemarkName1
function:        myFunc
loc:             myfile.mlir:12:3
args:
  - Remark:          vectorized loop
  - tripCount:       128
```

**Bitstream format** – compact binary for large runs.

***

### 2. **With `mlir::emitRemarks` (No Streamer)**

If the streamer isn't passed, the remarks are mirrored to the `DiagnosticEngine`
using `mlir::emitRemarks`

```c++
mlir::remark::RemarkCategories cats{/*passed=*/categoryLoopunroll,
                                     /*missed=*/std::nullopt,
                                     /*analysis=*/std::nullopt,
                                     /*failed=*/categoryLoopunroll};
remark::enableOptimizationRemarks(
    /*streamer=*/nullptr, cats,
    /*printAsEmitRemarks=*/true);
```

***

### 3. **With a Custom Streamer**

You can implement a custom streamer by inheriting `MLIRRemarkStreamerBase` to
consume remarks in any format.

```c++
class MyStreamer : public MLIRRemarkStreamerBase {
public:
  void streamOptimizationRemark(const Remark &remark) override {
    // Convert and write remark to your custom format
  }
};

auto myStreamer = std::make_unique<MyStreamer>();
remark::enableOptimizationRemarks(
    /*streamer=*/myStreamer, cats,
    /*printAsEmitRemarks=*/true);
```
