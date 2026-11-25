//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def WebAssemblyOptions : OptionsStruct<prefix = "wasm-">;

// WebAssemblyCFGSort.cpp
// Disabling is only for testing unwind destination mismatches in CFGStackify.
defm : BoolField<"wasm-ehpad-sort", "true",
  "WebAssembly: EH pad-first sort order">;
// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"wasm-disable-ehpad-sort", "wasm-ehpad-sort=false">;

// WebAssemblyLowerEmscriptenEHSjLj.cpp
defm : ListField<"emscripten-cxx-exceptions-allowed", "StringRef",
  "The list of function names in which Emscripten-style exception handling is enabled (see emscripten EMSCRIPTEN_CATCHING_ALLOWED options)">;

// WebAssemblyMCInstLower.cpp
// This disables the removal of registers when lowering into MC, as required by
// some current tests.
defm : BoolField<"wasm-keep-registers", "false",
  "WebAssembly: output stack registers in instruction output for test purposes only">;

// WebAssemblyPeephole.cpp
defm : BoolField<"wasm-fallthrough-return-opt", "true",
  "WebAssembly: Fallthrough-return optimizations">;
// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"disable-wasm-fallthrough-return-opt", "wasm-fallthrough-return-opt=false">;

// WebAssemblyTargetMachine.cpp
// Emscripten's asm.js-style setjmp/longjmp handling
defm : BoolField<"enable-emscripten-sjlj", "false",
  "WebAssembly Emscripten-style setjmp/longjmp handling">;
// Keep implicit locals for the purpose of testing with lit/llc ONLY. This
// produces output which is not valid WebAssembly, and is not supported by
// assemblers/disassemblers and other MC based tools.
defm : BoolField<"wasm-emit-explicit-locals", "true",
  "WebAssembly: explicit locals in instruction output (disable for test purposes only)">;
// TODO: Remove the old spellings after 2027-05.
def : AliasOf<"wasm-disable-explicit-locals", "wasm-emit-explicit-locals=false">;
// setjmp/longjmp handling using wasm EH instructions
defm : BoolField<"wasm-enable-sjlj", "false",
  "WebAssembly setjmp/longjmp handling">;
// If true, use the legacy Wasm EH proposal:
// https://github.com/WebAssembly/exception-handling/blob/main/proposals/exception-handling/legacy/Exceptions.md
// And if false, use the standardized Wasm EH proposal:
// https://github.com/WebAssembly/exception-handling/blob/main/proposals/exception-handling/Exceptions.md
// Currently set to true by default because not all major web browsers turn on
// the new standard proposal by default, but will later change to false.
defm : BoolField<"wasm-use-legacy-eh", "true",
  "WebAssembly exception handling (legacy)">;
