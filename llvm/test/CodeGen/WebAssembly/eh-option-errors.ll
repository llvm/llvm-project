target triple = "wasm32-unknown-unknown"

; RUN: not --crash llc < %s -enable-emscripten-sjlj -wasm-enable-sjlj 2>&1 | FileCheck %s --check-prefix=EM_SJLJ_W_WASM_SJLJ
; EM_SJLJ_W_WASM_SJLJ: LLVM ERROR: -enable-emscripten-sjlj not allowed with -wasm-enable-sjlj

; RUN: not --crash llc < %s -exception-model=emscripten -wasm-enable-sjlj 2>&1 | FileCheck %s --check-prefix=EM_EH_W_WASM_SJLJ
; EM_EH_W_WASM_SJLJ: LLVM ERROR: -exception-model=emscripten not allowed with -wasm-enable-sjlj

; RUN: not --crash llc < %s -exception-model=dwarf 2>&1 | FileCheck %s --check-prefix=EH_MODEL_DWARF
; EH_MODEL_DWARF: LLVM ERROR: -exception-model should be either 'none', 'wasm', or 'emscripten'
