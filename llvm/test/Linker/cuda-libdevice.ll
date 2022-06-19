; Prepare bitcode files.
; RUN: rm -rf %t && mkdir -p %t
; RUN: llvm-as %s -o %t/main.bc
; RUN: llvm-as %p/Inputs/libdevice-cuda-9.ll -o %t/libdevice.compute_35.10.bc
; RUN: llvm-as %p/Inputs/libdevice-cuda-10.ll -o %t/libdevice.10.bc
; RUN: llvm-as %p/Inputs/libdevice-cuda-11.ll -o %t/libdevice.11.10.bc
; RUN: llvm-as %p/Inputs/libdevice-cuda-9.ll -o %t/correct-libdevice-wrong-filename.bc
; RUN: llvm-as %p/Inputs/not-a-libdevice.ll -o %t/libdevice-with-wrong-info.bc

; No warnings expected when we link with libdevice variants
; RUN: llvm-link %t/main.bc %t/libdevice.compute_35.10.bc -S 2>&1 \
; RUN:  | FileCheck --check-prefixes COMMON,NOWARN %s
; RUN: llvm-link %t/main.bc %t/libdevice.10.bc -S 2>&1 \
; RUN:  | FileCheck --check-prefixes COMMON,NOWARN %s
; RUN: llvm-link %t/main.bc %t/libdevice.11.10.bc -S 2>&1 \
; RUN:  | FileCheck --check-prefixes COMMON,NOWARN %s

; But make sure we still issue warnings if we see unexpected filename, or
; unexpected triple or datalayout within a libdevice filename.
; RUN: llvm-link %t/main.bc %t/correct-libdevice-wrong-filename.bc -S 2>&1 \
; RUN:  | FileCheck --check-prefixes COMMON,WARN-TRIPLE %s
; RUN: llvm-link %t/main.bc %t/libdevice-with-wrong-info.bc -S 2>&1 \
; RUN:  | FileCheck --check-prefixes COMMON,WARN-TRIPLE,WARN-DL %s


target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"

; WARN-TRIPLE-DAG: warning: Linking two modules of different target triples:
; WARN-DL-DAG: warning: Linking two modules of different data layouts:

; NOWARN-NOT: warning:
; COMMON-DAG: target triple = "nvptx64-nvidia-cuda"
; NOWARN-NOT: warning:
