// RUN: not llvm-mc -triple aarch64-linux-gnu %s 2> %t > /dev/null
// RUN: FileCheck %s < %t

    .cpu invalid
    // CHECK: error: unknown CPU name

    .cpu generic+wibble+nowobble
    // CHECK: :[[@LINE-1]]:18: error: unsupported architectural extension
    // CHECK: :[[@LINE-2]]:25: error: unsupported architectural extension
