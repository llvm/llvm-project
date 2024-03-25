; RUN: not opt --mtriple powerpc64-unknown-aix < %s -passes=embed-bitcode -S 2>&1 | FileCheck %s

@a = global i32 1

; CHECK: LLVM ERROR: EmbedBitcode pass currently only supports ELF object format
