// RUN: not --crash llvm-mc -filetype=obj -triple x86_64-pc-linux %s -o %t 2>&1 | FileCheck %s

// CHECK:  sleb128 and uleb128 expressions must be absolute

        .sleb128 undefined
