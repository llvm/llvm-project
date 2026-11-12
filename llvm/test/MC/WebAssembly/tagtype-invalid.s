# RUN: not llvm-mc -triple=wasm32 %s 2>&1 | FileCheck %s

# CHECK: [[#@LINE+1]]:10: error: Expected identifier, got: 42
.tagtype 42

# CHECK: [[#@LINE+1]]:13: error: Expected EOL, instead got: ,
.tagtype foo, i32

# CHECK: [[#@LINE+1]]:18: error: Expected EOL, instead got: pub
.tagtype bar i32 pub
