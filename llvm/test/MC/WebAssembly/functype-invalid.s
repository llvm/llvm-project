# RUN: not llvm-mc -triple=wasm32 %s 2>&1 | FileCheck %s

# CHECK: [[#@LINE+1]]:10: error: Expected identifier, got:
.functype

# CHECK: [[#@LINE+1]]:13: error: Expected (, instead got:
.functype fn

# CHECK: [[#@LINE+1]]:15: error: Expected ), instead got:
.functype fn (

# CHECK: [[#@LINE+1]]:15: error: unknown type: i42
.functype fn (i42

# CHECK: [[#@LINE+1]]:19: error: Expected ), instead got: i32
.functype fn (i32 i32

# CHECK: [[#@LINE+1]]:16: error: Expected ->, instead got:
.functype fn ()

# CHECK: [[#@LINE+1]]:17: error: Expected ->, instead got: <
.functype fn () <- ()

# CHECK: [[#@LINE+1]]:21: error: Expected ), instead got:
.functype fn () -> (

# CHECK: [[#@LINE+1]]:23: error: Expected EOL, instead got: ->
.functype fn () -> () -> ()
