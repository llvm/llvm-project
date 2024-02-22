(module
  (type (;0;) (func (result i32)))
  (import "env" "__linear_memory" (memory (;0;) i64 1))
  (import "env" "__stack_pointer" (global (;0;) (mut i64)))
  (func $_Z4testv (type 0) (result i32)
    (local i32)
    i32.const 1
    local.set 0
    local.get 0
    return)
  (data $S (i64.const 0) "\00"))
