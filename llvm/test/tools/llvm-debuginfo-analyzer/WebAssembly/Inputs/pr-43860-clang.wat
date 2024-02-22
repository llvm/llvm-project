(module
  (type (;0;) (func (param i32 i32) (result i32)))
  (import "env" "__linear_memory" (memory (;0;) i64 0))
  (import "env" "__stack_pointer" (global (;0;) (mut i64)))
  (func $_Z4testii (type 0) (param i32 i32) (result i32)
    (local i64 i64 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
    global.get 0
    local.set 2
    i64.const 32
    local.set 3
    local.get 2
    local.get 3
    i64.sub
    local.set 4
    local.get 4
    local.get 0
    i32.store offset=16
    local.get 4
    local.get 1
    i32.store offset=12
    local.get 4
    i32.load offset=16
    local.set 5
    local.get 4
    local.get 5
    i32.store offset=8
    local.get 4
    i32.load offset=12
    local.set 6
    local.get 4
    local.get 6
    i32.store offset=28
    local.get 4
    i32.load offset=28
    local.set 7
    local.get 4
    local.get 7
    i32.store offset=24
    local.get 4
    i32.load offset=28
    local.set 8
    local.get 4
    i32.load offset=24
    local.set 9
    local.get 8
    local.get 9
    i32.add
    local.set 10
    local.get 4
    local.get 10
    i32.store offset=20
    local.get 4
    i32.load offset=20
    local.set 11
    local.get 4
    local.get 11
    i32.store offset=24
    local.get 4
    i32.load offset=24
    local.set 12
    local.get 4
    i32.load offset=8
    local.set 13
    local.get 13
    local.get 12
    i32.add
    local.set 14
    local.get 4
    local.get 14
    i32.store offset=8
    local.get 4
    i32.load offset=8
    local.set 15
    local.get 15
    return))
