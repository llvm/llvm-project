(module
  (type (;0;) (func (result i32)))
  (type (;1;) (func (param i64 i64) (result i32)))
  (type (;2;) (func (param i32 i64) (result i32)))
  (import "env" "__linear_memory" (memory (;0;) i64 1))
  (import "env" "__stack_pointer" (global (;0;) (mut i64)))
  (import "env" "_Z6printfPKcz" (func (;0;) (type 1)))
  (import "env" "__indirect_function_table" (table (;0;) 0 funcref))
  (func $__original_main (type 0) (result i32)
    (local i64 i64 i64 i32 i64 i64 i32 i64 i64)
    global.get 0
    local.set 0
    i64.const 16
    local.set 1
    local.get 0
    local.get 1
    i64.sub
    local.set 2
    local.get 2
    global.set 0
    i32.const 0
    local.set 3
    local.get 2
    local.get 3
    i32.store offset=12
    i64.const 0
    local.set 4
    i64.const 0
    local.set 5
    local.get 4
    local.get 5
    call 0
    drop
    i32.const 0
    local.set 6
    i64.const 16
    local.set 7
    local.get 2
    local.get 7
    i64.add
    local.set 8
    local.get 8
    global.set 0
    local.get 6
    return)
  (func $main (type 2) (param i32 i64) (result i32)
    (local i32)
    call $__original_main
    local.set 2
    local.get 2
    return)
  (data $.L.str (i64.const 0) "Hello, World\0a\00"))
