; RUN: rm -rf %t.cache
; RUN: opt -module-hash -module-summary %s -o %t.bc

; RUN: llvm-lto2 run -o %t.o %t.bc -cache-dir %t.cache -r=%t.bc,globalfunc,plx
; RUN: llvm-lto2 run -o %t.o %t.bc -cache-dir %t.cache -r=%t.bc,globalfunc,plx -mcpu=core2
; RUN: llvm-lto2 run -o %t.o %t.bc -cache-dir %t.cache -r=%t.bc,globalfunc,plx -relax-elf-relocations
; RUN: llvm-lto2 run -o %t.o %t.bc -cache-dir %t.cache -r=%t.bc,globalfunc,plx -function-sections
; RUN: llvm-lto2 run -o %t.o %t.bc -cache-dir %t.cache -r=%t.bc,globalfunc,plx -data-sections
; RUN: llvm-lto2 run -o %t.o %t.bc -cache-dir %t.cache -r=%t.bc,globalfunc,plx -debugger-tune=sce
; RUN: llvm-lto2 run -o %t.o %t.bc -cache-dir %t.cache -r=%t.bc,globalfunc,plx -mattr=+sse2
; RUN: llvm-lto2 run -o %t.o %t.bc -cache-dir %t.cache -r=%t.bc,globalfunc,plx -relocation-model=static
; RUN: llvm-lto2 run -o %t.o %t.bc -cache-dir %t.cache -r=%t.bc,globalfunc,plx -code-model=large
; RUN: llvm-lto2 run -o %t.o %t.bc -cache-dir %t.cache -r=%t.bc,globalfunc,plx -cg-opt-level=0
; RUN: llvm-lto2 run -o %t.o %t.bc -cache-dir %t.cache -r=%t.bc,globalfunc,plx -O1
; RUN: llvm-lto2 run -o %t.o %t.bc -cache-dir %t.cache -r=%t.bc,globalfunc,plx -opt-pipeline=loweratomic
; RUN: llvm-lto2 run -o %t.o %t.bc -cache-dir %t.cache -r=%t.bc,globalfunc,plx -aa-pipeline=basic-aa
; RUN: llvm-lto2 run -o %t.o %t.bc -cache-dir %t.cache -r=%t.bc,globalfunc,plx -override-triple=x86_64-unknown-linux-gnu
; RUN: llvm-lto2 run -o %t.o %t.bc -cache-dir %t.cache -r=%t.bc,globalfunc,plx -default-triple=x86_64-unknown-linux-gnu
; RUN: ls %t.cache | count 15

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @globalfunc() {
entry:
  ret void
}
