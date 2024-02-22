LLVM WebAssembly limitations
----------------------------
LLVM does not support any conversion between the binary format (.wasm)
and the text format (.wat):

  .wasm -> .wat
  .wat  -> .wasm

WebAssembly external tools
--------------------------
These tools are intended for use in (or for development of) toolchains
or other systems that want to manipulate WebAssembly files.

  Development of WebAssembly and associated infrastructure
  https://github.com/WebAssembly

  WABT: The WebAssembly Binary Toolkit
  https://github.com/WebAssembly/wabt

  wasm2wat — translate from the binary format to the text format
  https://webassembly.github.io/wabt/doc/wasm2wat.1.html

  wat2wasm — translate from WebAssembly text format to the WebAssembly binary format
  https://webassembly.github.io/wabt/doc/wat2wasm.1.html

How to generate .wasm from .cpp
-------------------------------
Each test includes its C++ source. To generate the .wasm for any of
the tests (ie. hello-world.cpp):

For target=wasm64
  clang --target=wasm64 -c -w -g -O0 -o hello-world.wasm hello-world.cpp

or

For target=wasm32
  clang --target=wasm32 -c -w -g -O0 -o hello-world.wasm hello-world.cpp

How to generate .wasm from .wat
-------------------------------
Using the 'wasm2wat' and 'wat2wasm' (WABT: The WebAssembly Binary Toolkit),
we can generate the WebAssembly text format file.

For target=wasm64
  wasm2wat hello-world.wasm --enable-memory64 -o hello-world.wat

For target=wasm32
  wat2wasm hello-world.wat -o hello-world.wasm

Notes:
------
* The tests run the .wasm file (target=wasm64).
* The .wat files were generated using WABT wasm2wat tool and are included
  only for extra information.
