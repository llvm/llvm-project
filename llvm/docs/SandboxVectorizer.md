# The Sandbox Vectorizer

```{contents}
:depth: 4
```

The Sandbox Vectorizer is a framework for building modular vectorization pipelines on top of [Sandbox IR](#SandboxIR) transactional IR, with a focus on ease of development and testing.
The default pipeline currently implements a simple SLP-style bottom-up vectorization pipeline.

The transactional IR helps in several ways:
- It enables a modular design where:
  - Each vectorization transformation/optimization can be implemented as a separate internal pass that uses actual IR as its input and output.
  - You can still make end-to-end profitability decisions (i.e., across multiple internal passes), even when the transformations are implemented as separate internal passes.
  - Each transformation/optimization internal pass can be tested in isolation with lit-tests, as opposed to end-to-end tests.
- It enables a simpler design by enabling each internal pass commit its state to the IR itself rather than updating helper data-structures that live across the pipeline.
- Its extensive callback interface helps remove the burden of manually maintaining the vectorizer's components while the IR is being modified.

## Usage

The Sandbox Vectorizer is currently under development and is not enabled by default.
So in order to use it you have to explicitly run the pass with `opt` like so:

```shell
$ opt -p=sandbox-vectorizer file.ll
```

## Internal Pass Pipeline

The Sandbox Vectorizer is designed to be modular and as such it has its own internal pass-pipeline that operates on Sandbox IR.
Each vectorization phase is implemented as a separate internal pass that runs by the Sandbox Vectorizer's internal pass manager.
The Sandbox Vectorizer pass itself is an LLVM Function pass.

The following figure shows the basic structure of the Sandbox Vectorizer LLVM Function pass.
The first component is the conversion of `LLVM IR to Sandbox IR` which converts the LLVM Function to a `sandboxir::Function`.
From this point on the pass operates on Sandbox IR.
The main entry point to the internal pass pipeline is the `Sandbox IR Function Pass Manger`, which runs all registered function passes.
The following figure lists only a single Sandbox IR function pass, the `Seed Collection Pass` which goes over the instructions in the function and collects vectorization candidates, like Stores to consecutive memory addresses, and forms a [Region](#region).
The `Seed Collection Pass` itself contains its own Region pass pipeline, which in the following example contains a `Transaction Save` pass, a `Bottom-Up Vectorization` pass, a `Pack Reuse` pass and a `Transaction Accept/Revert` pass.

```
┌────────────────────────────────── Sandbox Vectorizer LLVM Function Pass ─────────────────────────────┐
│                                                                                                      │
│ ┌───────┐ ┌────────────────────────── sandboxir::Function Pass Manager ────────────────────────────┐ │
│ │       │ │                                                                                        │ │
│ │       │ │ ┌────────────────────────────── Seed Collection Pass ──────────────────────────────┐   │ │
│ │       │ │ │                                                                                  │   │ │
│ │       │ │ │ ┌───────┐  For   ┌─────────────── sanboxir::Region Pass Manager ───────────────┐ │   │ │
│ │LLVM IR│ │ │ │Collect│  each  │ ┌───────────┐ ┌────────────────┐ ┌───────┐ ┌──────────────┐ │ │   │ │
│ │  to   │ │ │ │ Seeds │ Region │ │Transaction│ │   Bottom─Up    │ │ Pack  │ │ Transaction  │ │ │   │ │
│ │Sandbox│ │ │ │Create │ ─────> │ │   Save    │ │ Vectorization  │ │ Reuse │ │Accept/Revert │ │ │   │ │
│ │  IR   │ │ │ │Regions│        │ └───────────┘ └────────────────┘ └───────┘ └──────────────┘ │ │   │ │
│ │       │ │ │ └───────┘        └─────────────────────────────────────────────────────────────┘ │   │ │
│ │       │ │ │                                                                                  │...│ │
│ │       │ │ └──────────────────────────────────────────────────────────────────────────────────┘   │ │
│ │       │ │                                                                                        │ │
│ └───────┘ └────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

You can specify your own custom pipeline with the `-sbvec-passes=` argument to `opt`.
The pipeline shown above is equivalent to this:

```shell
$ opt -p=sandbox-vectorizer -sbvec-passes='seed-collection<tr-save,bottom-up-vec,pack-reuse,tr-accept>' file.ll
```

If the user does not define a pipeline, the Sandbox Vectorizer will run its default pass-pipeline, which is set in the constructor of the `SandboxVectorizerPass`.

## Sandbox Vectorizer Passes

The passes in the vectorization pipeline can be found in `Transforms/Vectorize/SandboxVectorizer/Passes` and they are registered in `lib/Transforms/Vectorize/SandboxVectorizer/Passes/PassRegistry.def`.

There are two types of passes: [Transformation Passes](#transformation-passes) that do the actual vectorization-related transformations and optimizations, and [Helper Passes](#helper-passes) that are helping with things like managing the IR transactions, and test-specific things like building regions.

### Transformation Passes

|  **Pass Name**            |         **File Name**       | **Type** |                     **Description**                     |
|---------------------------|-----------------------------|----------|---------------------------------------------------------|
| `seed-collection`         | SeedCollection.h            | Function | Collects the instructions to start vectorizing from, creates a region and runs the region-pass pipeline |
| `bottom-up-vec`           | BottomUpVec.h               | Region   | An SLP-style bottom-up vectorizer. It can vectorize both scalars and vectors |
| `pack-reuse`              | PackReuse.h                 | Region   | A pass that de-duplicates packs                         |

### Helper Passes

|  **Pass Name**            |         **File Name**       | **Type** |                     **Description**                     |
|---------------------------|-----------------------------|----------|---------------------------------------------------------|
| `tr-save`                 | TransactionSave.h           | Region   | Creates a checkpoint of the IR (i.e., saves state)      |
| `tr-accept`               | TransactionAlwaysAccept.h   | Region   | Unconditionally accepts the IR state                    |
| `tr-revert`               | TransactionAlwaysRevert.h   | Region   | Unconditionally rejects the IR state                    |
| `tr-accept-or-revert`     | TransactionAcceptOrRevert.h | Region   | Checks cost model and either accepts or reverts the IR  |
| `null`                    | NullPass.h                  | Region   | A dummy test pass that just returns                     |
| `print-region`            | PrintRegion.h               | Region   | A test pass that prints the region's IR                 |
| `print-instruction-count` | PrintInstructionCount.h     | Region   | A test pass that counts instructions                    |
| `regions-from-metadata`   | RegionsFromMetadata.h       | Function | Builds regions from IR metadata and runs a pipeline of region passes for each one of them. Used in lit tests for testing region passes in isolation |
| `regions-from-bbs`        | RegionsFromBBs.h            | Function | Builds a region for each BB, adding all BB instructions into each region. Used in lit tests for stress-testing region passes in isolation |

## Region

In a traditional compiler pass pipeline, transformations usually operate at a function level with function passes.
This introduces an issue in passes like the vectorizer that operate on small sections of a function (that we refer to as "Regions") but apply a pipeline of transformations on each section horizontally, and evaluate profitability end-to-end on each region as shown below:

```
 Function
┌─────────┐    Transform    Transform  ...    Transform
│         │        A            B                 Z
│┌───────┐│    ┌───────┐    ┌───────┐         ┌───────┐     Evaluate
││Region1││ ─> │       │ ─> │       │  ... ─> │       │   Profitability
│└───────┘│    └───────┘    └───────┘         └───────┘
│         │
│┌───────┐│    ┌───────┐    ┌───────┐         ┌───────┐     Evaluate
││Region2││ ─> │       │ ─> │       │  ... ─> │       │   Profitability
│└───────┘│    └───────┘    └───────┘         └───────┘
│         │
│         │
└─────────┘
```

If transformations like `A`, `B`, etc. are implemented as function passes, then they will apply their transformations across the whole function, spanning multiple regions, as they have not been designed to stay within a region.
The problem is that profitability evaluation will average out the profitability across all regions within the function, leading to a sub-optimal outcome.

This is the problem that the "Region" structure is solving.
It provides a way of tagging the instructions within a Region with metadata and also provides the necessary APIs for iterating through the Region instructions and operating on them.

The Region allows us to implement the vectorization pipeline as a pipeline of Region passes, each one operating on a specific code section.
At the end of the region pass pipeline we can evaluate profitability across multiple region passes in the pipeline (if needed) but within a Region, and either accept or revert the transformations.

### Adding Instructions to the Region

The Region grows automatically and is maintained transparently:
Whenever you create a new instruction it is automatically added to the Region, and whenever an instruction is deleted it gets removed from the Region.
The reasoning is that vectorization passes work: (i) by creating new vector instructions, (ii) by adding necessary packing/unpacking instructions, or (iii) by deleting the original instructions that got replaced by the vectorized ones.

Internally this is done with the help of the callback API of Sandbox IR.
The current Region gets notified that either an instruction got created or removed and the Region is maintained accordingly.

### Region Example

The following example defines a Region (with `!0 = distinct !{!"sandboxregion"}`), containing two instructions: `%i1 = add i8 %v, 1` and `%i2 = add i8 %v, 2` in no particular order.

```llvm
   define void @region_example(i8 %v) {
     %i0 = add i8 %v, 0
     %i1 = add i8 %v, 1, !sandboxvec !0
     %i2 = add i8 %v, 2, !sandboxvec !0
     ret void
   }
   !0 = distinct !{!"sandboxregion"}
```

The Region class API allows you to iterate through the region instructions like with a range loop:

```c++
   for (auto *I : Rgn)
     // Do something with `I`
```

### Region Auxiliary Vector

On top of tagging instructions the Region has a second functionality: it also supports a way of defining an ordered list of instructions.
This helps passes communicate such instruction lists from one pass to another, if needed, in an explicit way that is encoded in IR metadata.
This removes the need for sharing helper data-structures across passes.
The end result is that you can fully describe such ordered list of instructions in IR and can reproduce the pass behavior using just IR as input, allowing you to test it with lit tests.

The Region API for the auxiliary vector is straightforward.
It provides the `getAux()` getter method that simply returns the auxiliary vector.

The auxiliary vector instructions are marked with `!sandboxaux` followed by an index, which in the following example are `!1`and `!2` which correspond to 0 and 1 respectively.
So the following example defines one region (region `!0`) containing all three `add` instructions, two of which belong to the auxiliary vector: `[%i1 = add i8 %v, 1, %i2 = add i8 %v, 2]`.

```llvm
   define void @region_aux_example(i8 %v) {
     %i0 = add i8 %v, 0  !sandboxvec !0
     %i1 = add i8 %v, 1, !sandboxvec !0, !sandboxaux !1
     %i2 = add i8 %v, 2, !sandboxvec !0, !sandboxaux !2
     ret void
   }
   !0 = distinct !{!"sandboxregion"}
   !1 = !{i32 0}
   !2 = !{i32 1}
```

The auxiliary vector is currently used by the Seed Collection pass to communicate a group of seed instructions to the Bottom-Up-Vectorizer pass.

## Testing Sandbox Vectorizer Passes In Isolation

One of the great things about the Sandbox Vectorizer is that it allows you to test each internal pass in isolation with lit-tests.

Testing Function passes is straightforward, just run `FUNCTION_PASS` in isolation with `-sbvec-passes`, like so:
```shell
$ opt -p=sandbox-vectorizer -sbvec-passes='FUNCTION_PASS'
```

Testing [Region](#region) passes is also possible, since a Region can be defined with IR using metadata, as described in [Region Example](#region-example).
We need to run the `regions-from-metadata` helper pass before the Region pass to be tested.
This helper pass parses the IR metadata looking for Region metadata, then it creates the corresponding Regions and finally runs a Region pass pipeline on each Region.

So here is what we need to do for a working lit-test of a Region pass:

1. Define the region with metadata as explained in [Region Example](#region-example).
2. Define the pass pipeline which should include the following passes:
   - The `regions-from-metadata` pass that will form a region and will build a Region pass-manager.
   - The `REGION_PASS` being tested enclosed in `< >`, as the only pass in the region pass pipeline owned by the `regions-from-metadata` pass.

So overall the pipeline looks like:
```shell
$ opt -p=sandbox-vectorizer -sbvec-passes='regions-from-metadata<REGION_PASS>'
```

The reason for enclosing the pass in `< >` is that `regions-from-metadata` is a Region pass manager function pass that accepts string arguments within `< >`.
It will parse the string argument, looking for a comma-separated list of Region pass names, and will populate the pipeline with these passes.
So in this case `REGION_PASS` is the only pass name found, so it will be the only pass added to the region pass pipeline.

For example, `'regions-from-metadata<region_pass1,region_pass2>'` would create regions from metadata, and for each one of them it would run the pipeline: `region_pass1` followed by `region_pass2`.

## Stress-testing a Region Pass in Isolation

A region pass can be stress-tested in isolation using with BB-sized Regions, using the `regions-from-bbs` helper pass.
The pass will go through the BBs in the function, create a region including all BB instructions for each one of them.
Then it will run the region pass pipeline for each BB-sized Region.

For example:
```shell
$ opt -p=sandbox-vectorizer -sbvec-passes='regions-from-bbs<REGION_PASS>'
```

## Components

The Sandbox Vectorizer implements several components that are used by one or more internal passes.
These components are designed as standalone components, which makes them easy to use when needed.
They are the building blocks for the vectorization passes, providing things like vectorization legality checks.

### Legality Analysis

This is the main entry point for vectorization legality checks.
It checks if a bundle of instructions is legal to vectorize and returns how the vectorizer should generate code for it, i.e., whether it should trivially widen it, whether it should reuse an existing vector, whether it should pack scalars into vectors and so on.
Legality testing includes tests like checking the instruction types, the opcodes etc. but it also checks for dependency violations by querying the [Scheduler](#scheduler).

The main API function is `LegalityAnalysis::canVectorize()`.

### Scheduler

This component is an lazy list-scheduler that relies on a [Dependency Graph (DAG)](#dependency-graph) for representing the dependencies.
It is "lazy" in the sense that it does not operate on a whole BB, but instead only on the instructions spanning the bundle we are attempting to schedule.
The main interface to the scheduler is `Scheduler::trySchedule()`.

Please note that the current implementation does not use a separate "staging" instruction list for the scheduled instructions.
Instead it physically moves the instructions in the IR chain, which is fine since we are working with a transactional IR.
This is not a requirement though for correct operation.
A more traditional scheduler with a separate instruction list would also work fine.

### Dependency Graph

This is a lazily-built Directed Acyclic Graph (DAG) that encodes both memory and def-use dependencies.
Each node of the graph points to a Sandbox IR Instruction.
An edge between two nodes `A->B` suggests that `A`'s instruction should come before `B`'s instruction in the program.
The DAG uses [Alias Analysis](#AliasAnalysis) for finding the memory dependencies.
Note that even though the DAG Node provides an API for iterating over both memory and use-def dependencies, it actually relies on the LLVM IR use-def edges internally and won't replicate them to save memory.

The graph is built lazily, meaning that it won't be built for the whole BB in one go.
Instead it will span only the range of instruction needed by the scheduler.
As we keep scheduling more instructions, the graph will grow on-demand following the needs of the scheduler.
The main interface function for the Dependency Graph is `DependencyGraph::extend()`.

### InstrMaps

Instruction Maps is a helper data structure that maintains a mapping between the original (often scalar) instructions and their corresponding vector instructions and the reverse.
It is used by the `bottom-up-vec` region pass for tracing vector instructions back to the original instructions and the reverse.


## Debugging

There are a couple of useful `cl::opt` options for debugging the vectorizer, that are particularly useful for bisection debugging:

| **Option**                      | **Description**                                    |
|---------------------------------|----------------------------------------------------|
| `-sbvec-allow-files=<regex>`    | Enables the Sandbox Vectorizer as a whole only for source files matching the comma-separated list of regular expressions. |
| `-sbvec-passes=<pass-pipeline>` | Allows you to change the internal pass pipeline and skip any potentially broken passes. |
| `-sbvec-stop-at=<num>`          | Will stop invoking the bottom-up-vectorizer if the invocation count is greater or equal to `<num>`. |
| `-sbvec-stop-bndl=<num>`        | Limits the vectorization depth of the bottom-up vectorizer to `<num>`. This means that the vectorizer will emit a pack and stop vectorizing further. |
