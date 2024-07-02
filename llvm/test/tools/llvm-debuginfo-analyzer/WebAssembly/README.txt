Notes:
------
As we should avoid committing binaries (.wasm) to be used in tests,
instead we provide the '.cpp' source files and the '.s' files.

- For the tests, only the '.s' files are required.
- We use the target 'wasm32' as the 'wasm64' is not standardized yet.

How to generate .s from .cpp
----------------------------
Use clang to generate the '.s'.

  clang --target=wasm32 -S -g Inputs/hello-world.cpp -o Inputs/hello-world-clang.s
  clang --target=wasm32 -S -g Inputs/pr-43860.cpp    -o Inputs/pr-43860-clang.s
  clang --target=wasm32 -S -g Inputs/pr-44884.cpp    -o Inputs/pr-44884-clang.s
  clang --target=wasm32 -S -g Inputs/pr-46466.cpp    -o Inputs/pr-46466-clang.s
  clang --target=wasm32 -S -g Inputs/test.cpp        -o Inputs/test-clang.s

How to generate .o from .s
--------------------------------
Each test executes one of the following commands in order to generate
the binary '.wasm' used by that specific test:

  llvm-mc -arch=wasm32 -filetype=obj %p/Inputs/hello-world-clang.s -o hello-world-clang.o
  llvm-mc -arch=wasm32 -filetype=obj %p/Inputs/pr-43860-clang.s    -o pr-43860-clang.o
  llvm-mc -arch=wasm32 -filetype=obj %p/Inputs/pr-44884-clang.s    -o pr-44884-clang.o
  llvm-mc -arch=wasm32 -filetype=obj %p/Inputs/pr-46466-clang.s    -o pr-46466-clang.o
  llvm-mc -arch=wasm32 -filetype=obj %p/Inputs/test-clang.s        -o test-clang.o
