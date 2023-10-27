# RUN: not llvm-mc -triple=wasm32 %s 2>&1 | FileCheck %s

# CHECK: [[#@LINE+1]]:16: error: Expected identifier, got: 42
.import_module 42

# CHECK: [[#@LINE+1]]:19: error: Expected ,, instead got:
.import_module foo

# CHECK: [[#@LINE+1]]:20: error: Expected identifier, got:
.import_module foo,

# CHECK: [[#@LINE+1]]:24: error: Expected EOL, instead got: ,
.import_module foo, bar,
