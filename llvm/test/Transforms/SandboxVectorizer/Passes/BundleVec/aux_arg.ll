; RUN: not opt -passes=sandbox-vectorizer -sbvec-passes="seed-collection<bundle-vec>" %s -disable-output 2>&1 | FileCheck %s
; RUN: not opt -passes=sandbox-vectorizer -sbvec-passes="seed-collection<bundle-vec(sandman)>" %s -disable-output 2>&1 | FileCheck %s
; RUN: opt -passes=sandbox-vectorizer -sbvec-passes="seed-collection<bundle-vec(bottom-up)>" %s -disable-output
; RUN: opt -passes=sandbox-vectorizer -sbvec-passes="seed-collection<bundle-vec(top-down)>" %s -disable-output

; The direction argument is mandatory: neither a missing nor an unrecognized
; argument is accepted.

; CHECK: LLVM ERROR: bundle-vec requires either 'bottom-up' or 'top-down' as its aux argument!

define void @aux_arg(ptr %ptr) {
  ret void
}
