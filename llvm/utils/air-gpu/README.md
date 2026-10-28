# AIR hardware tests

`air_gpu_suite.py` runs kernels compiled by the AIR target on a real Apple GPU
and compares every result with a reference computed in Python.
`air_gpu_run.py` runs one AIR library with given buffers, for lit tests.

Both talk to the AIR service: a JSON file queue at `/air` (override with
`--queue` or `AIR_QUEUE`) served by a Mac, with the client in `/air/client`
(override with `AIR_CLIENT`). The service accepts AIR bitcode and runs it; it
refuses Metal source and compiler flags.

```
python3 air_gpu_suite.py --llc build/bin/llc                 # every group
python3 air_gpu_suite.py --llc build/bin/llc --group llvm-int --json out.json
```

## No Apple compiler output

Every kernel is generated from the text of `air_gpu_suite.py` and compiled by
this repository's AIR target. Do not add `.air` or `.metallib` files produced
by Apple's Metal compiler, and do not use them as references: that includes
the fixtures under `Metal.jl/test/metallib`, which are built from `.metal`
sources with Apple's toolchain. References are computed from the definition
of each operation.

## Results

* `PASS`: the device ran the kernel and every result matches.
* `MISMATCH`: some result is wrong (beyond the case's ULP or absolute bound).
* `HW-FTZ`: the device ran the kernel, and every wrong result is the
  hardware flushing float32 subnormals to zero (Apple GPUs do; LLVM's default
  is IEEE subnormals). Strict: each wrong thread must have a subnormal float32
  input or a subnormal exact result, and its value must equal the reference
  computed from inputs with float32 subnormals flushed, the result flushed too
  (the device value is compared flushed). Anything else is `MISMATCH`. A case
  can opt out with `ftz_ok = False`. Like `PASS`, `ABSENT` and `HOST-LIMIT`,
  it does not fail the run. Half subnormals are not flushed.
* `REJECTED`: the device refused the library. A group is first compiled as one
  library; when that is refused, each case is retried alone, so a rejection
  names the one kernel that causes it.
* `ABSENT`: an expected rejection, for a probe of a function AIR does not
  have (for example `air.copysign`).
* `COMPILE`: `llc` failed.

The service sometimes fails with `XPC_ERROR_CONNECTION_INTERRUPTED`; requests
are retried a few times. A kernel that fails that way on every retry has
crashed the Metal compiler service, and is reported as `REJECTED`.
