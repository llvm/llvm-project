; RUN: split-file %s %t

; RUN: not llc < %t/noflag.ll -enable-emscripten-sjlj -wasm-enable-sjlj 2>&1 | FileCheck %s --check-prefix=EM_SJLJ_W_WASM_SJLJ
; EM_SJLJ_W_WASM_SJLJ: LLVM ERROR: -enable-emscripten-sjlj not allowed with -wasm-enable-sjlj

; RUN: not llc < %t/noflag.ll -exception-model=emscripten -wasm-enable-sjlj 2>&1 | FileCheck %s --check-prefix=EM_EH_W_WASM_SJLJ
; RUN: not opt -passes=wasm-lower-em-ehsjlj -wasm-enable-sjlj -disable-output %t/emscripten.ll 2>&1 | FileCheck %s --check-prefix=EM_EH_W_WASM_SJLJ
; EM_EH_W_WASM_SJLJ: LLVM ERROR: 'emscripten' exception model is not allowed with -wasm-enable-sjlj

; RUN: not llc < %t/noflag.ll -exception-model=wasm -enable-emscripten-sjlj 2>&1 | FileCheck %s --check-prefix=WASM_EH_W_EM_SJLJ
; RUN: not opt -passes=wasm-lower-em-ehsjlj -enable-emscripten-sjlj -disable-output %t/wasm.ll 2>&1 | FileCheck %s --check-prefix=WASM_EH_W_EM_SJLJ
; WASM_EH_W_EM_SJLJ: LLVM ERROR: 'wasm' exception model is not allowed with -wasm-enable-sjlj

; RUN: not llc < %t/noflag.ll -exception-model=dwarf 2>&1 | FileCheck %s --check-prefix=EH_MODEL_DWARF
; RUN: not opt -passes=wasm-lower-em-ehsjlj -disable-output %t/dwarf.ll 2>&1 | FileCheck %s --check-prefix=EH_MODEL_DWARF
; EH_MODEL_DWARF: LLVM ERROR: exception model should be 'none', 'wasm', or 'emscripten'

;--- noflag.ll
target triple = "wasm32-unknown-unknown"

;--- emscripten.ll
target triple = "wasm32-unknown-emscripten"

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"emscripten"}

;--- wasm.ll
target triple = "wasm32-unknown-unknown"

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"wasm"}

;--- dwarf.ll
target triple = "wasm32-unknown-unknown"

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"dwarf"}
