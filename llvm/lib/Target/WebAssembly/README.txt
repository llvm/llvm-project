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
  https://build.chromium.org/p/client.wasm.llvm/console

The backend's bringup is done using the GCC torture test suite first since it
doesn't require C library support. Current known failures are in
known_gcc_test_failures.txt, all other tests should pass. The waterfall will
turn red if not. Once most of these pass, further testing will use LLVM's own
test suite.

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

Load and store instructions can have a constant offset. We should (a) model
this, and (b) do address-mode folding with it.

//===---------------------------------------------------------------------===//

Br, br_if, and tableswitch instructions can support having a value on the
expression stack across the jump (sometimes). We should (a) model this, and
(b) extend the stackifier to utilize it.

//===---------------------------------------------------------------------===//
