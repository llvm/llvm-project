(amdgpu-async-operations)=

# AMDGPU Asynchronous Operations


## Introduction

Asynchronous operations are operations whose completion is not tracked
internally by the compiler. A thread that initiates one or more async operations can use
*asyncmarks* to track their completion.

- Most {ref}`DMA operations <amdgpu-dma-operations>` are asynchronous.

## Asyncmarks

An *asyncmark* created by a thread can be used to track async operations
initiated by that thread.

### Stages

Every asyncmark belongs to a *stage*, which names a kind of async operation.
Each async operation has an *own stage*, determined by the instruction that
initiates it, and *belongs to* that stage and to the stage `ALL`.

The stages are:

| Value | Stage | Async operations |
|---|---|---|
| 0 | `TENSOR` | tensor loads and stores |
| 1 | `GLOBAL_LOAD_ASYNC_TO_LDS` | global loads async to LDS |
| 2 | `GLOBAL_LOAD_ASYNC_TO_LDS_MCAST` | multicast (cluster) global loads async to LDS |
| 3 | `ASYNC_LDS_STORE` | async stores from LDS |
| 5 | `UNFORMATTED_BUFFER_GLOBAL_LOAD` | unformatted buffer and global loads to LDS |
| 16 | `ALL` | all of the above |

The values not listed above are reserved for future async operations.
Using one is an error.

Which async operations a given subtarget actually has is described in
{ref}`AMDGPU DMA Operations <amdgpu-dma-operations>`. A stage is valid on every
subtarget that supports asyncmarks, whether or not that subtarget has any
operation belonging to it; marking an empty stage is simply a no-op.

Stage `ALL` is the catch-all: since every async operation belongs to it, an
asyncmark in `ALL` tracks all async operations whatever their own stage.

### Current Sequence

The abstract machine maintains a separate sequence of asyncmarks *for each
stage* during the execution of a function body, which excludes any asyncmarks
produced by calls to other functions encountered in the currently executing
function. The state of the sequence for a stage `S` at each program point in
the function is called the *current sequence of* `S`.

The sequences of distinct stages are independent: appending to one does not
affect the length or contents of any other.

### `@llvm.amdgcn.asyncmark(i32 %S)`

Produces an asyncmark in stage `S` and appends it to the current sequence of
`S`. `S` must be a constant naming a stage that is not reserved.

### `@llvm.amdgcn.wait.asyncmark(i16 %N, i32 %S)`

Ensures that the length of the current sequence of `S` is at most `N` by
removing asyncmarks from the start of that sequence if it is more than `N`. The
sequences of other stages are unaffected. `S` must be a constant naming a stage
that is not reserved.

### Completion of Asyncmarks

An `asyncmark()` operation `X` that produces an asyncmark `M` in stage `S` is
*completed-at* a `wait.asyncmark()` operation `Y` on stage `S` in the same
function body if:

- `X` is *program-ordered* before `Y`, and
- `M` is not in the current sequence of `S` at any operation `Z` that
  immediately follows `Y` in *program-order*.

## Completion of Async Operations

An async operation executes outside the thread that initiated it, i.e., it is
not related in *program-order* with any other operations from that thread. But
the thread can use an asyncmark to ensure that the async operation is
*completed-at* some later operation.

An async operation `A` *initiated-by* an instruction `I` is *completed-at* some
`wait.asyncmark()` operation `Y` if there exists an `asyncmark()` operation `X`
in a stage `S` such that:
- `A` belongs to `S`,
- `I` is *program-ordered* before `X`, and
- `X` is *completed-at* `Y`.

Since `A` belongs to `ALL` as well as to its own stage, an asyncmark and wait
pair on `ALL` tracks `A` regardless of its own stage.

### happens-before

When an instruction `I` initiates an async operation `A`, `I` *happens-before*
`A`.

If `A` is *completed-at* a `wait.asyncmark()` operation `Y`, then `A`
*happens-before* `Y`.

## Examples

The stage argument is omitted in the examples below; they all use stage `ALL`,
so every asyncmark tracks every async operation and there is only one sequence
to reason about. The {ref}`interleaved stages <amdgpu-async-interleaved-stages>`
example shows what stages add.

### Uneven blocks of async operations

```c++
void foo(global int *g, local int *l) {
  // first block
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  asyncmark();

  // second block; longer
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  asyncmark();

  // third block; shorter
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  asyncmark();

  // Wait for first block
  wait.asyncmark(2);
}
```

### Software pipeline

```c++
void foo(global int *g, local int *l) {
  // first block
  asyncmark();

  // second block
  asyncmark();

  // third block
  asyncmark();

  for (;;) {
    wait.asyncmark(2);
    // use data

    // next block
    asyncmark();
  }

  // flush one block
  wait.asyncmark(2);

  // flush one more block
  wait.asyncmark(1);

  // flush last block
  wait.asyncmark(0);
}
```

### Ordinary function call

```c++
extern void bar(); // may or may not initiate async operations

void foo(global int *g, local int *l) {
    // first block
    asyncmark();

    // second block
    asyncmark();

    // function call
    bar();

    // third block
    asyncmark();

    // wait for the second block
    wait.asyncmark(1);

    // wait for the third block, including bar()
    wait.asyncmark(0);
}
```

(amdgpu-async-interleaved-stages)=
### Interleaved stages

Two kinds of async operation are in flight at once. Each is marked in its own
stage, so each can be waited for without waiting for the other.

```c++
void foo(global int *g, local int *l, tensor_desc t) {
  // Start a long tensor load and mark it in the TENSOR stage.
  tensor_load_to_lds(l, t);
  asyncmark(TENSOR);

  // Start a short LDS load and mark it in its own stage.
  async_load_to_lds(l, g);
  asyncmark(GLOBAL_LOAD_ASYNC_TO_LDS);

  // Wait for the LDS load only. The tensor load is still in flight: its
  // asyncmark is in a different sequence, so this wait neither counts nor
  // removes it.
  wait.asyncmark(0, GLOBAL_LOAD_ASYNC_TO_LDS);

  // perform synchronization / use the data loaded by async_load_to_lds

  // Now wait for the tensor load.
  wait.asyncmark(0, TENSOR);
}
```

Had both marks been placed in `ALL`, the first wait would have drained the
tensor load as well, and the overlap would have been lost.

The same applies to two stages tracked by one hardware counter. Here both
stages use the async counter, but the sequences are still separate:

```c++
void foo(global int *g, local int *l) {
  async_load_to_lds(l, g);
  asyncmark(GLOBAL_LOAD_ASYNC_TO_LDS);   // X

  async_store_from_lds(g, l);
  asyncmark(ASYNC_LDS_STORE);            // Y

  // Completes X. Y is in a different sequence and is not completed here, even
  // though both stages are tracked by the same counter.
  wait.asyncmark(0, GLOBAL_LOAD_ASYNC_TO_LDS);
}
```

## Implementation notes

[This section is informational.]

### Function Calls

In general, at a function call, if the caller uses sufficient waits to track
its own async operations, the actions performed by the callee cannot affect
correctness. But inlining such a call may result in redundant waits.

```c++
void foo() {
  ...
  asyncmark();       // X
  ...                // no wait.asyncmark()
}

void bar() {
  asyncmark();       // B
  asyncmark();       // C
  foo();
  wait.asyncmark(1); // D
}
```

Before inlining, it is unspecified whether `X` is *completed-at* `D`, while
`C` is **not** *completed-at* `D`. The programmer can only rely on `B`
being *completed-at* `D`.

```c++
void bar() {
  asyncmark();       // B
  asyncmark();       // C
  ...
  asyncmark();       // X
  ...                // no wait.asyncmark()
  wait.asyncmark(1); // D
}
```

After inlining, `C` is also *completed-at* `D` and `X` is **not**
*completed-at* `D`.

Conversely, a `wait.asyncmark` call inside a callee cannot be used to track
asyncmarks from the caller, since this `wait.asyncmark` can only
observe the current sequence of the callee.

```c++
void foo() {
  ...                // no asyncmark()
  wait.asyncmark(0); // Y
  ...
}

void bar() {
  asyncmark();       // B
  asyncmark();       // C
  foo();
  wait.asyncmark(1); // D
}
```

In the above example, it is unspecified whether `B` and `C` in `bar()` are
*completed-at* `Y`, because they are not included in the sequence that can be
examined at `Y`.

```c++
void bar() {
  asyncmark();       // B
  asyncmark();       // C
  ...                // no asyncmark()
  wait.asyncmark(0); // Y
  ...
  wait.asyncmark(1); // D
}
```

After inlining, both `B` and `C` are *completed-at* `Y`.

### Optimization

The implementation may eliminate asyncmark/wait intrinsics in the following
cases. These are just examples and not meant to be an exhaustive list. Each
applies per stage: the sequences are independent, so an asyncmark in one stage
is unaffected by the waits of another.

1. An `asyncmark` operation in a stage `S` which remains in the current
   sequence of `S` along every path that reaches the function exit.

   ```c++
   void foo() {
     ...
     asyncmark();       // X
     ...                // no wait.asyncmark()
   }
   ```

   Here, `X` can be eliminated.

2. A `wait.asyncmark` on a stage `S` which sees an empty sequence of asyncmarks
   for `S` along every path that reaches it.

   ```c++
   void foo() {
     ...                // no asyncmark()
     wait.asyncmark(0); // Y
     ...
   }
   ```

   Here, `Y` can be eliminated.
