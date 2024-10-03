# RUN: not llvm-mc -triple=wasm32 -mattr=+exception-handling,+reference-types,+tail-call %s 2>&1 | FileCheck %s

# These tests are intended to act as a litmus test for the WebAssembly ASM
# type-checker - both in terms of errors it can catch and in terms of the
# location information used in the error messages.

local_get_no_local_type:
  .functype local_get_no_local_type () -> ()
# CHECK: :[[@LINE+1]]:13: error: no local type specified for index 0
  local.get 0
  end_function

local_set_no_local_type:
  .functype local_set_no_local_type () -> ()
# CHECK: :[[@LINE+1]]:13: error: no local type specified for index 0
  local.set 0
  end_function

local_set_empty_stack_while_popping:
  .functype local_set_empty_stack_while_popping () -> ()
  .local i32
# CHECK: [[@LINE+1]]:3: error: type mismatch, expected [i32] but got []
  local.set 0
  end_function

local_set_type_mismatch:
  .functype local_set_type_mismatch () -> ()
  .local i32
  f32.const 1.0
# CHECK: [[@LINE+1]]:3: error: type mismatch, expected [i32] but got [f32]
  local.set 0
  end_function

local_tee_no_local_type:
  .functype local_tee_no_local_type () -> ()
# CHECK: :[[@LINE+1]]:13: error: no local type specified for index 0
  local.tee 0
  end_function

local_tee_empty_stack_while_popping:
  .functype local_tee_empty_stack_while_popping () -> ()
  .local f32
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [f32] but got []
  local.tee 0
  end_function

local_tee_type_mismatch:
  .functype local_tee_type_mismatch () -> ()
  .local f32
  i32.const 1
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [f32] but got [i32]
  local.tee 0
  drop
  end_function

global_get_missing_globaltype:
  .functype global_get_missing_globaltype () -> ()
# CHECK: :[[@LINE+1]]:14: error: symbol foo: missing .globaltype
  global.get foo
  end_function

global_get_expected_expression_operand:
  .functype global_get_expected_expression_operand () -> ()
# CHECK: :[[@LINE+1]]:14: error: expected expression operand
  global.get 1
  end_function

global_set_missing_globaltype:
  .functype global_set_missing_globaltype () -> ()
# CHECK: :[[@LINE+1]]:14: error: symbol foo: missing .globaltype
  global.set foo
  end_function

global_set_expected_expression_operand:
  .functype global_set_expected_expression_operand () -> ()
# CHECK: :[[@LINE+1]]:14: error: expected expression operand
  global.set 1
  end_function

global_set_empty_stack_while_popping:
  .functype global_set_empty_stack_while_popping () -> ()
  .globaltype valid_global, i64
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i64] but got []
  global.set valid_global
  end_function

global_set_type_mismatch:
  .functype global_set_type_mismatch () -> ()
  .globaltype valid_global, i64
  i32.const 1
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i64] but got [i32]
  global.set valid_global
  end_function

table_get_expected_expression_operand:
  .functype table_get_expected_expression_operand () -> ()
  i32.const 0
# CHECK: :[[@LINE+1]]:13: error: expected expression operand
  table.get 1
  end_function

table_get_missing_tabletype:
  .functype table_get_missing_tabletype () -> ()
  i32.const 0
# CHECK: :[[@LINE+1]]:13: error: symbol foo: missing .tabletype
  table.get foo
  end_function

.tabletype valid_table, externref

table_get_empty_stack_while_popping:
  .functype table_get_empty_stack_while_popping () -> ()
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got []
  table.get valid_table
  drop
  end_function

table_get_type_mismatch:
  .functype table_get_type_mismatch () -> ()
  f32.const 1.0
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got [f32]
  table.get valid_table
  drop
  end_function

table_set_expected_expression_operand:
  .functype table_set_expected_expression_operand () -> ()
  i32.const 0
# CHECK: :[[@LINE+1]]:13: error: expected expression operand
  table.set 1
  end_function

table_set_missing_tabletype:
  .functype table_set_missing_tabletype () -> ()
  i32.const 0
# CHECK: :[[@LINE+1]]:13: error: symbol foo: missing .tabletype
  table.set foo
  end_function

table_set_empty_stack_while_popping_1:
  .functype table_set_empty_stack_while_popping_1 () -> ()
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32, externref] but got []
  table.set valid_table
  end_function

table_set_empty_stack_while_popping_2:
  .functype table_set_empty_stack_while_popping_2 (externref) -> ()
  local.get 0
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32, externref] but got [externref]
  table.set valid_table
  end_function

table_set_type_mismatch_1:
  .functype table_set_type_mismatch_1 () -> ()
  i32.const 0
  ref.null_func
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32, externref] but got [i32, funcref]
  table.set valid_table
  end_function

table_set_type_mismatch_2:
  .functype table_set_type_mismatch_2 () -> ()
  f32.const 1.0
  ref.null_extern
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32, externref] but got [f32, externref]
  table.set valid_table
  end_function

table_fill_expected_expression_operand:
  .functype table_fill_expected_expression_operand () -> ()
  i32.const 0
  ref.null_extern
  i32.const 4
# CHECK: :[[@LINE+1]]:14: error: expected expression operand
  table.fill 1
  end_function

table_fill_missing_tabletype:
  .functype table_fill_missing_tabletype () -> ()
  i32.const 0
  ref.null_extern
  i32.const 4
# CHECK: :[[@LINE+1]]:14: error: symbol foo: missing .tabletype
  table.fill foo
  end_function

table_fill_empty_stack_while_popping_1:
  .functype table_fill_empty_stack_while_popping_1 () -> ()
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32, externref, i32] but got []
  table.fill valid_table
  end_function

table_fill_empty_stack_while_popping_2:
  .functype table_fill_empty_stack_while_popping_2 (i32) -> ()
  local.get 0
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32, externref, i32] but got [i32]
  table.fill valid_table
  end_function

table_fill_empty_stack_while_popping_3:
  .functype table_fill_empty_stack_while_popping_3 (i32, externref) -> ()
  local.get 1
  local.get 0
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32, externref, i32] but got [externref, i32]
  table.fill valid_table
  end_function

table_fill_type_mismatch_1:
  .functype table_fill_type_mismatch_1 () -> ()
  i32.const 0
  ref.null_extern
  ref.null_func
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32, externref, i32] but got [i32, externref, funcref]
  table.fill valid_table
  end_function

table_fill_type_mismatch_2:
  .functype table_fill_type_mismatch_2 () -> ()
  i32.const 0
  ref.null_func
  i32.const 1
# CHECK: [[@LINE+1]]:3: error: type mismatch, expected [i32, externref, i32] but got [i32, funcref, i32]
  table.fill valid_table
  end_function

table_fill_type_mismatch_3:
  .functype table_fill_type_mismatch_3 () -> ()
  f32.const 2.0
  ref.null_extern
  i32.const 1
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32, externref, i32] but got [f32, externref, i32]
  table.fill valid_table
  end_function

table_fill_type_mismatch_4:
  .functype table_fill_type_mismatch_4 () -> ()
  i32.const 1
  ref.null_exn
  i32.const 1
# CHECK: [[@LINE+1]]:3: error: type mismatch, expected [i32, externref, i32] but got [i32, exnref, i32]
  table.fill valid_table
  end_function

table_grow_non_exist_table:
  .functype table_grow_non_exist_table (externref, i32) -> (i32)
  local.get 0
  local.get 1
# CHECK: [[@LINE+1]]:14: error: symbol invalid_table: missing .tabletype
  table.grow invalid_table
  end_function

table_grow_type_mismatch_1:
  .functype table_grow_type_mismatch_1 (externref, i32) -> (i32)
  local.get 1
# CHECK: [[@LINE+1]]:3: error: type mismatch, expected [externref, i32] but got [i32]
  table.grow valid_table
  end_function

table_grow_type_mismatch_2:
  .functype table_grow_type_mismatch_2 (externref, i32) -> (i32)
  local.get 0
  local.get 0
# CHECK: [[@LINE+1]]:3: error: type mismatch, expected [externref, i32] but got [externref, externref]
  table.grow valid_table
  end_function

table_grow_wrong_result:
  .functype table_grow_wrong_result (externref, i32) -> (f32)
  local.get 0
  local.get 1
  table.grow valid_table
# CHECK: [[@LINE+1]]:3: error: type mismatch, expected [f32] but got [i32]
  end_function

drop_empty_stack_while_popping:
  .functype drop_empty_stack_while_popping () -> ()
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [any] but got []
  drop
  end_function

end_block_insufficient_values_on_stack_1:
  .functype end_block_insufficient_values_on_stack_1 () -> ()
  block i32
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got []
  end_block
  drop
  end_function

end_block_insufficient_values_on_stack_2:
  .functype end_block_insufficient_values_on_stack_2 () -> ()
  block () -> (i32)
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got []
  end_block
  drop
  end_function

end_block_type_mismatch:
  .functype end_block_type_mismatch () -> ()
  block i32
  f32.const 1.0
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got [f32]
  end_block
  drop
  end_function

end_loop_insufficient_values_on_stack:
  .functype end_loop_insufficient_values_on_stack () -> ()
  loop i32
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got []
  end_loop
  drop
  end_function

end_loop_type_mismatch:
  .functype end_loop_type_mismatch () -> ()
  loop f32
  i32.const 1
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [f32] but got [i32]
  end_loop
  drop
  end_function

end_if_insufficient_values_on_stack_1:
  .functype end_if_insufficient_values_on_stack_1 () -> ()
  i32.const 1
  if i32
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got []
  end_if
  end_function

end_if_type_mismatch_1:
  .functype end_if_type_mismatch_1 () -> ()
  i32.const 1
  if f32
  i32.const 1
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [f32] but got [i32]
  end_if
  drop
  end_function

end_if_insufficient_values_on_stack_2:
  .functype end_if_insufficient_values_on_stack_2 () -> ()
  i32.const 1
  if i32
  i32.const 2
  else
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got []
  end_if
  drop
  end_function

end_if_type_mismatch_2:
  .functype end_if_type_mismatch_2 () -> ()
  i32.const 1
  if i32
  i32.const 2
  else
  f32.const 3.0
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got [f32]
  end_if
  drop
  end_function

else_insufficient_values_on_stack:
  .functype else_insufficient_values_on_stack () -> ()
  i32.const 1
  if i32
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got []
  else
  i32.const 0
  end_if
  drop
  end_function

else_type_mismatch:
  .functype else_type_mismatch () -> ()
  i32.const 1
  if i32
  f32.const 0.0
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got [f32]
  else
  i32.const 0
  end_if
  drop
  end_function

.tagtype tag_i32 i32
.tagtype tag_f32 f32

end_try_insufficient_values_on_stack:
  .functype end_try_insufficient_values_on_stack () -> ()
  try i32
  i32.const 0
  catch_all
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got []
  end_try
  drop
  end_function

end_try_type_mismatch:
  .functype end_try_type_mismatch () -> ()
  try i32
  i32.const 0
  catch tag_f32
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got [f32]
  end_try
  drop
  end_function

catch_insufficient_values_on_stack:
  .functype catch_insufficient_values_on_stack () -> ()
  try i32
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got []
  catch tag_i32
  end_try
  drop
  end_function

catch_type_mismatch:
  .functype catch_type_mismatch () -> ()
  try i32
  f32.const 1.0
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got [f32]
  catch tag_i32
  end_try
  drop
  end_function

catch_all_insufficient_values_on_stack:
  .functype catch_all_insufficient_values_on_stack () -> ()
  try i32
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got []
  catch_all
  i32.const 0
  end_try
  drop
  end_function

catch_all_type_mismatch:
  .functype catch_all_type_mismatch () -> ()
  try i32
  f32.const 1.0
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got [f32]
  catch_all
  i32.const 0
  end_try
  drop
  end_function

delegate_insufficient_values_on_stack:
  .functype delegate_insufficient_values_on_stack () -> ()
  try i32
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got []
  delegate 0
  drop
  end_function

delegate_type_mismatch:
  .functype delegate_type_mismatch () -> ()
  try i32
  f32.const 1.0
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got [f32]
  delegate 0
  drop
  end_function

end_function_empty_stack_while_popping:
  .functype end_function_empty_stack_while_popping () -> (i32)
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got []
  end_function

end_function_type_mismatch:
  .functype end_function_type_mismatch () -> (f32)
  i32.const 1
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [f32] but got [i32]
  end_function

end_function_superfluous_end_function_values:
  .functype end_function_superfluous_end_function_values () -> ()
  i32.const 1
  f32.const 2.0
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [] but got [i32, f32]
  end_function

return_empty_stack_while_popping:
  .functype return_empty_stack_while_popping () -> (i32)
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got []
  return
  end_function

return_type_mismatch:
  .functype return_type_mismatch () -> (f32)
  i32.const 1
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [f32] but got [i32]
  return
  end_function

# Missing index for call_indirect.
call_indirect_empty_stack_while_popping_1:
  .functype call_indirect_empty_stack_while_popping_1 () -> ()
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got []
  call_indirect () -> ()
  end_function

# Missing arguments for target of call_indirect.
call_indirect_empty_stack_while_popping_2:
  .functype call_indirect_empty_stack_while_popping_1 (f32) -> ()
  i32.const 1
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [f32] but got []
  call_indirect (f32) -> ()
  end_function

call_indirect_type_mismatch_for_argument:
  .functype call_indirect_type_mismatch_for_argument () -> ()
  i32.const 1
  i32.const 2
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [f32] but got [i32]
  call_indirect (f32) -> ()
  end_function

call_indirect_superfluous_value_at_end:
  .functype call_indirect_superfluous_value_at_end () -> ()
  i32.const 1
  call_indirect () -> (i64)
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [] but got [i64]
  end_function

# Missing index for return_call_indirect.
return_call_indirect_empty_stack_while_popping_1:
  .functype return_call_indirect_empty_stack_while_popping_1 () -> ()
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got []
  return_call_indirect () -> ()
  end_function

# Missing arguments for target of return_call_indirect.
return_call_indirect_empty_stack_while_popping_2:
  .functype return_call_indirect_empty_stack_while_popping_2 () -> ()
  i32.const 1
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [f32] but got []
  return_call_indirect (f32) -> ()
  end_function

call_expected_expression_operand:
  .functype call_expected_expression_operand () -> ()
# CHECK: :[[@LINE+1]]:8: error: expected expression operand
  call 1
  end_function

.functype fn_i32_to_void (i32) -> ()

call_empty_stack_while_popping:
  .functype call_empty_stack_while_popping () -> ()
# CHECK: [[@LINE+1]]:3: error: type mismatch, expected [i32] but got []
  call fn_i32_to_void
  end_function

call_type_mismatch:
  .functype call_type_mismatch () -> ()
  f32.const 1.0
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got [f32]
  call fn_i32_to_void
  end_function

.functype fn_void_to_i32 () -> (i32)

call_superfluous_value_at_end:
  .functype call_superfluous_value_at_end () -> ()
  call fn_void_to_i32
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [] but got [i32]
  end_function

call_missing_functype:
  .functype call_missing_functype () -> ()
# CHECK: :[[@LINE+1]]:8: error: symbol no_functype: missing .functype
  call no_functype
  end_function

return_call_expected_expression_operand:
  .functype return_call_expected_expression_operand () -> ()
# CHECK: :[[@LINE+1]]:15: error: expected expression operand
  return_call 1
  end_function

return_call_empty_stack_while_popping:
  .functype return_call_empty_stack_while_popping () -> ()
# CHECK: [[@LINE+1]]:3: error: type mismatch, expected [i32] but got []
  return_call fn_i32_to_void
  end_function

return_call_type_mismatch:
  .functype return_call_type_mismatch () -> ()
  f32.const 1.0
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got [f32]
  return_call fn_i32_to_void
  end_function

return_call_missing_functype:
  .functype return_call_missing_functype () -> ()
# CHECK: :[[@LINE+1]]:15: error: symbol no_functype: missing .functype
  return_call no_functype
  end_function

catch_expected_expression_operand:
  .functype catch_expected_expression_operand () -> ()
  try
# CHECK: :[[@LINE+1]]:9: error: expected expression operand
  catch 1
  end_try
  end_function

catch_missing_tagtype:
  .functype catch_missing_tagtype () -> ()
  try
# CHECK: :[[@LINE+1]]:9: error: symbol no_tagtype: missing .tagtype
  catch no_tagtype
  end_try
  end_function

catch_superfluous_value_at_end:
  .functype catch_superfluous_value_at_end () -> ()
  try
  catch tag_i32
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [] but got [i32]
  end_try
  end_function

ref_is_null_empty_stack_while_popping:
  .functype ref_is_null_empty_stack_while_popping () -> ()
# CHECK: [[@LINE+1]]:3: error: type mismatch, expected [ref] but got []
  ref.is_null
  drop
  end_function

ref_is_null_type_mismatch:
  .functype ref_is_null_type_mismatch () -> ()
  i32.const 1
# CHECK: [[@LINE+1]]:3: error: type mismatch, expected [ref] but got [i32]
  ref.is_null
  drop
  end_function

ref_is_null_pushes_i32:
  .functype ref_is_null_pushes_i32 () -> (i64)
  ref.null_func
  ref.is_null
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i64] but got [i32]
  end_function

# For the other instructions, the type checker checks vs the operands in the
# instruction definition. Perform some simple checks for these rather than
# exhaustively testing all instructions.

other_insn_test_1:
  .functype other_insn_test_1 () -> ()
# CHECK: [[@LINE+1]]:3: error: type mismatch, expected [i32, i32] but got []
  i32.add
  drop
  end_function

other_insn_test_2:
  .functype other_insn_test_2 () -> ()
  i32.const 1
  ref.null_func
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32, i32] but got [i32, funcref]
  i32.add
  drop
  end_function

other_insn_test_3:
  .functype other_insn_test_3 () -> ()
  f32.const 1.0
  f32.const 2.0
  f32.add
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [] but got [f32]
  end_function

# Unreachable code within 'block' does not affect type checking after
# 'end_block'
check_after_unreachable_within_block:
  .functype check_after_unreachable_within_block () -> ()
  block
  unreachable
  end_block
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [any] but got []
  drop
  end_function

# Unreachable code within 'loop' does not affect type checking after 'end_loop'
check_after_unreachable_within_loop:
  .functype check_after_unreachable_within_loop () -> ()
  loop
  unreachable
  end_loop
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [any] but got []
  drop
  end_function

# Unreachable code within 'if' does not affect type checking after 'end_if'
check_after_unreachable_within_if_1:
  .functype check_after_unreachable_within_if_1 () -> ()
  i32.const 0
  if
  unreachable
  else
  unreachable
  end_if
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [any] but got []
  drop
  end_function

# Unreachable code within 'if' does not affect type checking after 'else'
check_after_unreachable_within_if_2:
  .functype check_after_unreachable_within_if_2 () -> ()
  i32.const 0
  if
  unreachable
  else
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [any] but got []
  drop
  end_if
  end_function

# Unreachable code within 'try' does not affect type checking after 'end_try'
check_after_unreachable_within_try_1:
  .functype check_after_unreachable_within_try_1 () -> ()
  try
  unreachable
  catch_all
  unreachable
  end_try
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [any] but got []
  drop
  end_function

# Unreachable code within 'try' does not affect type checking after 'catch'
check_after_unreachable_within_try_2:
  .functype check_after_unreachable_within_try_2 () -> ()
  try
  unreachable
  catch tag_i32
  drop
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [any] but got []
  drop
  end_try
  end_function

# Unreachable code within 'try' does not affect type checking after 'catch_all'
check_after_unreachable_within_try_3:
  .functype check_after_unreachable_within_try_3 () -> ()
  try
  unreachable
  catch_all
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [any] but got []
  drop
  end_try
  end_function

# Unreachable code within 'try' does not affect type checking after 'delegate'
check_after_unreachable_within_try_4:
  .functype check_after_unreachable_within_try_4 () -> ()
  try
  unreachable
  delegate 0
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [any] but got []
  drop
  end_function

br_invalid_type_loop:
  .functype br_invalid_type_loop () -> ()
  i32.const 1
  loop (i32) -> (f32)
    drop
    f32.const 1.0
# CHECK: :[[@LINE+1]]:5: error: type mismatch, expected [i32] but got [f32]
    br 0
  end_loop
  drop
  end_function

br_invalid_type_block:
  .functype br_invalid_type_block () -> ()
  i32.const 1
  block (i32) -> (f32)
# CHECK: :[[@LINE+1]]:5: error: type mismatch, expected [f32] but got [i32]
    br 0
    f32.const 1.0
  end_block
  drop
  end_function

br_invalid_type_if:
  .functype br_invalid_type_if () -> ()
  i32.const 1
  if f32
    f32.const 1.0
  else
    i32.const 1
# CHECK: :[[@LINE+1]]:5: error: type mismatch, expected [f32] but got [i32]
    br 0
  end_if
  drop
  end_function

br_invalid_type_try:
  .functype br_invalid_type_try () -> ()
  try f32
    i32.const 1
# CHECK: :[[@LINE+1]]:5: error: type mismatch, expected [f32] but got [i32]
    br 0
  catch tag_f32
  end_try
  drop
  end_function

br_invalid_type_catch:
  .functype br_invalid_type_catch () -> ()
  try f32
    f32.const 1.0
  catch tag_i32
# CHECK: :[[@LINE+1]]:5: error: type mismatch, expected [f32] but got [i32]
    br 0
  end_try
  drop
  end_function

br_invalid_type_catch_all:
  .functype br_invalid_type_catch_all () -> ()
  try f32
    f32.const 1.0
  catch_all
    i32.const 1
# CHECK: :[[@LINE+1]]:5: error: type mismatch, expected [f32] but got [i32]
    br 0
  end_try
  drop
  end_function

br_invalid_depth_out_of_range:
  .functype br_invalid_depth_out_of_range () -> ()
  block
  block
  block
# CHECK: :[[@LINE+1]]:5: error: br: invalid depth 4
    br 4
  end_block
  end_block
  end_block
  end_function

br_incorrect_signature:
  .functype br_incorrect_signature () -> ()
  block f32
    block i32
      i32.const 1
# CHECK: :[[@LINE+1]]:7: error: type mismatch, expected [f32] but got [i32]
      br 1
    end_block
    drop
    f32.const 1.0
  end_block
  drop
  end_function

br_incorrect_func_signature:
  .functype br_incorrect_func_signature () -> (i32)
  block f32
    f32.const 1.0
# CHECK: :[[@LINE+1]]:5: error: type mismatch, expected [i32] but got [f32]
    br 1
  end_block
  drop
  i32.const 1
  end_function

multiple_errors_in_function:
  .functype multiple_errors_in_function () -> ()
# CHECK: :[[@LINE+2]]:3: error: type mismatch, expected [i32] but got []
# CHECK: :[[@LINE+1]]:13: error: expected expression operand
  table.get 1

# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32, externref, i32] but got [any]
  table.fill valid_table

  f32.const 0.0
  ref.null_extern
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32, i32] but got [f32, externref]
  i32.add
  drop
  end_function

.functype take_and_return_multi(i32, i64, f32, f64) -> (i32, i64, f32, f64)

call_with_multi_param_and_return:
  .functype call_with_multi_param_and_return () -> (i32)
  ref.null_extern
  f32.const 0.0
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32, i64, f32, f64] but got [externref, f32]
  call take_and_return_multi
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got [i32, i64, f32, f64]
  end_function

.functype callee (f32, i32) -> ()

any_value_on_stack:
  .functype any_value_on_stack () -> ()
  # This local does not exist so it should error out, but it should put an 'any'
  # value on the stack so 'call callee' should not error out again
# CHECK: :[[@LINE+1]]:13: error: no local type specified for index 0
  local.get 0
  i32.const 0
# CHECK-NOT: :[[@LINE+1]]:3: error: type mismatch
  call callee

  # But this time 'call callee' should error out
  i32.const 0
# CHECK: :[[@LINE+1]]:13: error: no local type specified for index 0
  local.get 0
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [f32, i32] but got [i32, any]
  call callee

# CHECK: :[[@LINE+2]]:13: error: no local type specified for index 0
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [any] but got []
  local.set 0
  drop

  end_function

block_param_and_return:
  .functype block_param_and_return () -> ()
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got []
  block (i32) -> (f32)
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [f32] but got [i32]
  end_block
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [i32] but got [f32]
  i32.popcnt
  drop

  block f32
  f32.const 0.0
  br 0
  i32.const 0
# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [f32] but got [..., i32]
  end_block

# CHECK: :[[@LINE+1]]:3: error: type mismatch, expected [] but got [f32]
  end_function
