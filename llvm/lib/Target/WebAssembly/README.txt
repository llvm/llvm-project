//===-- README.txt - Notes for WebAssembly code gen -----------------------===//

This WebAssembly backend is presently in a very early stage of development.
The code should build and not break anything else, but don't expect a lot more
at this point.

For more information on WebAssembly itself, see the design documents:
  * https://github.com/WebAssembly/design/blob/master/README.md

The following documents contain some information on the planned semantics and
binary encoding of WebAssembly itself:
  * https://github.com/WebAssembly/design/blob/master/AstSemantics.md
  * https://github.com/WebAssembly/design/blob/master/BinaryEncoding.md

The backend is built, tested and archived on the following waterfall:
  https://wasm-stat.us

The backend's bringup is done using the GCC torture test suite first since it
doesn't require C library support. Current known failures are in
known_gcc_test_failures.txt, all other tests should pass. The waterfall will
turn red if not. Once most of these pass, further testing will use LLVM's own
test suite. The tests can be run locally using:
  https://github.com/WebAssembly/waterfall/blob/master/src/compile_torture_tests.py

Interesting work that remains to be done:
* Write a pass to restructurize irreducible control flow. This needs to be done
  before register allocation to be efficient, because it may duplicate basic
  blocks and WebAssembly performs register allocation at a whole-function
  level. Note that LLVM's GPU code has such a pass, but it linearizes control
  flow (e.g. both sides of branches execute and are masked) which is undesirable
  for WebAssembly.

//===---------------------------------------------------------------------===//

set_local instructions have a return value. We should (a) model this,
and (b) write optimizations which take advantage of it. Keep in mind that
many set_local instructions are implicit!

//===---------------------------------------------------------------------===//

Br, br_if, and tableswitch instructions can support having a value on the
expression stack across the jump (sometimes). We should (a) model this, and
(b) extend the stackifier to utilize it.

//===---------------------------------------------------------------------===//

The min/max operators aren't exactly a<b?a:b because of NaN and negative zero
behavior. The ARM target has the same kind of min/max instructions and has
implemented optimizations for them; we should do similar optimizations for
WebAssembly.

//===---------------------------------------------------------------------===//

AArch64 runs SeparateConstOffsetFromGEPPass, followed by EarlyCSE and LICM.
Would these be useful to run for WebAssembly too? Also, it has an option to
run SimplifyCFG after running the AtomicExpand pass. Would this be useful for
us too?

//===---------------------------------------------------------------------===//

Register stackification uses the EXPR_STACK physical register to impose
ordering dependencies on instructions with stack operands. This is pessimistic;
we should consider alternate ways to model stack dependencies.

//===---------------------------------------------------------------------===//

Lots of things could be done in WebAssemblyTargetTransformInfo.cpp. Similarly,
there are numerous optimization-related hooks that can be overridden in
WebAssemblyTargetLowering.

//===---------------------------------------------------------------------===//

Instead of the OptimizeReturned pass, which should consider preserving the
"returned" attribute through to MachineInstrs and extending the StoreResults
pass to do this optimization on calls too. That would also let the
WebAssemblyPeephole pass clean up dead defs for such calls, as it does for
stores.

//===---------------------------------------------------------------------===//

Consider implementing optimizeSelect, optimizeCompareInstr, optimizeCondBranch,
optimizeLoadInstr, and/or getMachineCombinerPatterns.

//===---------------------------------------------------------------------===//

Find a clean way to fix the problem which leads to the Shrink Wrapping pass
being run after the WebAssembly PEI pass.

//===---------------------------------------------------------------------===//
