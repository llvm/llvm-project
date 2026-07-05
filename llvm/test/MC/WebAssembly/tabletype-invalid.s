# RUN: not llvm-mc -triple=wasm32 %s 2>&1 | FileCheck %s

# CHECK: [[#@LINE+1]]:11: error: Expected identifier, got:
.tabletype

# CHECK: [[#@LINE+1]]:12: error: Expected identifier, got: 42
.tabletype 42

# CHECK: [[#@LINE+1]]:15: error: Expected ,, instead got:
.tabletype sym

# CHECK: [[#@LINE+1]]:16: error: Expected identifier, got:
.tabletype sym,

# CHECK: [[#@LINE+1]]:17: error: Expected identifier, got: 42
.tabletype sym, 42

# CHECK: [[#@LINE+1]]:17: error: Unknown type in .tabletype directive: i42
.tabletype sym, i42

# CHECK: [[#@LINE+1]]:21: error: Expected integer constant, instead got:
.tabletype sym, i32,

# CHECK: [[#@LINE+1]]:25: error: Expected integer constant, instead got:
.tabletype sym, i32, 42,

# CHECK: [[#@LINE+1]]:28: error: Expected EOL, instead got: ,
.tabletype sym, i32, 42, 42,
