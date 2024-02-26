(module
  (type (;0;) (func (param f32) (result i32)))
  (type (;1;) (func (param i32) (result i32)))
  (import "env" "__linear_memory" (memory (;0;) 0))
  (import "env" "__stack_pointer" (global (;0;) (mut i32)))
  (func $_Z3barf (type 0) (param f32) (result i32)
    (local i32 i32 i32 f32 f32 f32 i32 i32 i32 i32 i32 i32)
    global.get 0
    local.set 1
    i32.const 16
    local.set 2
    local.get 1
    local.get 2
    i32.sub
    local.set 3
    local.get 3
    local.get 0
    f32.store offset=12
    local.get 3
    f32.load offset=12
    local.set 4
    local.get 4
    f32.abs
    local.set 5
    f32.const 0x1p+31 (;=2.14748e+09;)
    local.set 6
    local.get 5
    local.get 6
    f32.lt
    local.set 7
    local.get 7
    i32.eqz
    local.set 8
    block  ;; label = @1
      block  ;; label = @2
        local.get 8
        br_if 0 (;@2;)
        local.get 4
        i32.trunc_f32_s
        local.set 9
        local.get 9
        local.set 10
        br 1 (;@1;)
      end
      i32.const -2147483648
      local.set 11
      local.get 11
      local.set 10
    end
    local.get 10
    local.set 12
    local.get 12
    return)
  (func $_Z3fooc (type 1) (param i32) (result i32)
    (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 f32 f32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
    global.get 0
    local.set 1
    i32.const 16
    local.set 2
    local.get 1
    local.get 2
    i32.sub
    local.set 3
    local.get 3
    global.set 0
    local.get 3
    local.get 0
    i32.store8 offset=15
    local.get 3
    i32.load8_u offset=15
    local.set 4
    i32.const 24
    local.set 5
    local.get 4
    local.get 5
    i32.shl
    local.set 6
    local.get 6
    local.get 5
    i32.shr_s
    local.set 7
    local.get 3
    local.get 7
    i32.store offset=8
    local.get 3
    i32.load offset=8
    local.set 8
    local.get 3
    i32.load8_u offset=15
    local.set 9
    i32.const 24
    local.set 10
    local.get 9
    local.get 10
    i32.shl
    local.set 11
    local.get 11
    local.get 10
    i32.shr_s
    local.set 12
    local.get 8
    local.get 12
    i32.add
    local.set 13
    local.get 13
    f32.convert_i32_s
    local.set 14
    local.get 3
    local.get 14
    f32.store offset=4
    local.get 3
    f32.load offset=4
    local.set 15
    local.get 15
    call $_Z3barf
    local.set 16
    local.get 3
    local.get 16
    i32.store offset=8
    local.get 3
    i32.load offset=8
    local.set 17
    local.get 3
    i32.load8_u offset=15
    local.set 18
    i32.const 24
    local.set 19
    local.get 18
    local.get 19
    i32.shl
    local.set 20
    local.get 20
    local.get 19
    i32.shr_s
    local.set 21
    local.get 17
    local.get 21
    i32.add
    local.set 22
    i32.const 16
    local.set 23
    local.get 3
    local.get 23
    i32.add
    local.set 24
    local.get 24
    global.set 0
    local.get 22
    return))
