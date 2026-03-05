# RUN: not llvm-mc -triple=wasm32 %s 2>&1 | FileCheck %s

# CHECK: [[#@LINE+1]]:14: error: Expected identifier, got: 42
.import_name 42

# CHECK: [[#@LINE+1]]:17: error: Expected ,, instead got:
.import_name foo

# CHECK: [[#@LINE+1]]:18: error: Expected identifier, got:
.import_name foo,

# CHECK: [[#@LINE+1]]:22: error: Expected EOL, instead got: ,
.import_name foo, bar,
