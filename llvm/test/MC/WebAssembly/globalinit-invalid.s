# RUN: not llvm-mc -triple=wasm32 %s 2>&1 | FileCheck %s

# CHECK: error: .globalinit requires a preceding .globaltype for: nodecl
.globalinit nodecl, i32.const 1

# CHECK: error: Expected identifier, got: 42
.globalinit 42, i32.const 1

# A symbol of another kind cannot take an initial value.
.functype fn () -> ()
# CHECK: error: .globalinit is only valid for a global: fn
.globalinit fn, i32.const 1
.tabletype tbl, externref
# CHECK: error: .globalinit is only valid for a global: tbl
.globalinit tbl, i32.const 1
.tagtype tag i32
# CHECK: error: .globalinit is only valid for a global: tag
.globalinit tag, i32.const 1

.globaltype sym, i32
# CHECK: error: Expected a const instruction in .globalinit: hello
.globalinit sym, hello
# CHECK: error: Initial value does not match the global's type: f32.const
.globalinit sym, f32.const 1.0
# CHECK: error: Expected integer constant, instead got:
.globalinit sym, i32.const
# CHECK: error: Expected EOL, instead got: ,
.globalinit sym, i32.const 42,

# CHECK: error: Integer constant does not fit in the global's type: 4294967296
.globalinit sym, i32.const 4294967296
# CHECK: :[[@LINE+1]]:28: error: Integer constant does not fit in the global's type: -4294967295
.globalinit sym, i32.const -4294967295

.globaltype i64sym, i64
# CHECK: error: Integer constant does not fit in the global's type: 18446744073709551616
.globalinit i64sym, i64.const 18446744073709551616

.globaltype vec, v128
# CHECK: error: Initial value is not supported for this global type: v128.const
.globalinit vec, v128.const 0

.globaltype flt, f32
# CHECK: error: Expected floating-point constant, instead got: hello
.globalinit flt, f32.const hello
# CHECK: error: Expected floating-point constant, instead got: notanumber
.globalinit flt, f32.const notanumber
# The value takes the spellings an f32.const operand takes.
# CHECK: error: Cannot parse real: 0.1
.globalinit flt, f32.const 0.1
# CHECK: error: Expected floating-point constant, instead got: inf
.globalinit flt, f32.const inf
# CHECK: error: Expected floating-point constant, instead got: 1
.globalinit flt, f32.const 1

# A later .globaltype may repeat the type the value was given, but not
# contradict it.
.globaltype retyped, i32
.globalinit retyped, i32.const 42
.globaltype retyped, i32
# CHECK: error: Type does not match the .globalinit already given for: retyped
.globaltype retyped, f64
# CHECK: error: Type does not match the .globalinit already given for: retyped
.globaltype retyped, v128

# Repeating the same initial value is fine; contradicting it is not.
.globaltype conflict, i32
.globalinit conflict, i32.const 1
.globalinit conflict, i32.const 1
# CHECK: error: Conflicting .globalinit for: conflict
.globalinit conflict, i32.const 2

# nan and -nan are different values.
.globaltype nanconflict, f32
.globalinit nanconflict, f32.const nan
# CHECK: error: Conflicting .globalinit for: nanconflict
.globalinit nanconflict, f32.const -nan
