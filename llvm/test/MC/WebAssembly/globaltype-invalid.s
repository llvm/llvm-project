# RUN: not llvm-mc -triple=wasm32 %s 2>&1 | FileCheck %s

# CHECK: [[#@LINE+1]]:12: error: Expected identifier, got:
.globaltype

# CHECK: [[#@LINE+1]]:13: error: Expected identifier, got: 42
.globaltype 42

# CHECK: [[#@LINE+1]]:16: error: Expected ,, instead got:
.globaltype sym

# CHECK: [[#@LINE+1]]:17: error: Expected identifier, got:
.globaltype sym,

# CHECK: [[#@LINE+1]]:18: error: Expected identifier, got: 42
.globaltype sym, 42

# CHECK: [[#@LINE+1]]:18: error: Unknown type in .globaltype directive: i42
.globaltype sym, i42

# CHECK: [[#@LINE+1]]:22: error: Expected identifier, got:
.globaltype sym, i32,

# CHECK: [[#@LINE+1]]:23: error: Expected identifier, got: 42
.globaltype sym, i32, 42

# CHECK: [[#@LINE+1]]:23: error: Unknown type in .globaltype modifier: unmutable
.globaltype sym, i32, unmutable

# CHECK: [[#@LINE+1]]:32: error: Expected EOL, instead got: ,
.globaltype sym, i32, immutable,
