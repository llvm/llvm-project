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

Interesting work that remains to be done:
* Write a pass to restructurize irreducible control flow. This needs to be done
  before register allocation to be efficient, because it may duplicate basic
  blocks and WebAssembly performs register allocation at a whole-function
  level. Note that LLVM's GPU code has such a pass, but it linearizes control
  flow (e.g. both sides of branches execute and are masked) which is undesirable
  for WebAssembly.
* Basic relooper to expose control flow as an AST.
* Figure out how to properly use MC for virtual ISAs. This may require some
  refactoring of MC.

//===---------------------------------------------------------------------===//
