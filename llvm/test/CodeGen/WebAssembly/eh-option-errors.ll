target triple = "wasm32-unknown-unknown"

; RUN: not --crash llc < %s -exception-model=emscripten -wasm-enable-eh 2>&1 | FileCheck %s --check-prefix=EM_EH_W_WASM_EH
; EM_EH_W_WASM_EH: LLVM ERROR: -exception-model=emscripten not allowed with -wasm-enable-eh

; RUN: not --crash llc < %s -enable-emscripten-sjlj -wasm-enable-sjlj 2>&1 | FileCheck %s --check-prefix=EM_SJLJ_W_WASM_SJLJ
; EM_SJLJ_W_WASM_SJLJ: LLVM ERROR: -enable-emscripten-sjlj not allowed with -wasm-enable-sjlj

; RUN: not --crash llc < %s -exception-model=emscripten -wasm-enable-sjlj 2>&1 | FileCheck %s --check-prefix=EM_EH_W_WASM_SJLJ
; EM_EH_W_WASM_SJLJ: LLVM ERROR: -exception-model=emscripten not allowed with -wasm-enable-sjlj

; RUN: not --crash llc < %s -wasm-enable-eh -exception-model=dwarf 2>&1 | FileCheck %s --check-prefix=EH_MODEL_DWARF
; EH_MODEL_DWARF: LLVM ERROR: -exception-model should be either 'none', 'wasm', or 'emscripten'

; RUN: not --crash llc < %s -exception-model=wasm 2>&1 | FileCheck %s --check-prefix=MODEL_WASM_WO_WASM_EH_SJLJ
; MODEL_WASM_WO_WASM_EH_SJLJ: LLVM ERROR: -exception-model=wasm only allowed with at least one of -wasm-enable-eh or -wasm-enable-sjlj
