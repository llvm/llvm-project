// RUN: mlir-opt %s -split-input-file \
// RUN:    -async-to-async-runtime="eliminate-blocking-await-ops=true" \
// RUN:   | FileCheck %s --dump-input=always

// CHECK-LABEL: func @simple_callee
// CHECK-SAME: (%[[ARG:.*]]: f32)
// CHECK-SAME: -> (!async.token, !async.value<f32> {builtin.foo = "bar"})
func @simple_callee(%arg0: f32) -> (f32 {builtin.foo = "bar"}) {
// CHECK: %[[TOKEN:.*]] = async.runtime.create : !async.token
// CHECK: %[[RETURNED_STORAGE:.*]] = async.runtime.create : !async.value<f32>
// CHECK: %[[ID:.*]] = async.coro.id
// CHECK: %[[HDL:.*]] = async.coro.begin %[[ID]]

// CHECK: %[[VAL:.*]] = addf %[[ARG]], %[[ARG]] : f32
  %0 = addf %arg0, %arg0 : f32
// CHECK: %[[VAL_STORAGE:.*]] = async.runtime.create : !async.value<f32>
  %1 = async.runtime.create: !async.value<f32>
// CHECK: async.runtime.store %[[VAL]], %[[VAL_STORAGE]] : !async.value<f32>
  async.runtime.store %0, %1: !async.value<f32>
// CHECK: async.runtime.set_available %[[VAL_STORAGE]] : !async.value<f32>
  async.runtime.set_available %1: !async.value<f32>

// CHECK: %[[SAVED:.*]] = async.coro.save %[[HDL]]
// CHECK: async.runtime.await_and_resume %[[VAL_STORAGE]], %[[HDL]]
// CHECK: async.coro.suspend %[[SAVED]]
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME:.*]], ^[[CLEANUP:.*]]
  %2 = async.await %1 : !async.value<f32>

// CHECK: ^[[RESUME]]:
// CHECK:   %[[IS_ERROR:.*]] = async.runtime.is_error %[[VAL_STORAGE]] : !async.value<f32>
// CHECK:   cond_br %[[IS_ERROR]], ^[[BRANCH_ERROR:.*]], ^[[BRANCH_OK:.*]]

// CHECK: ^[[BRANCH_OK]]:
// CHECK:   %[[LOADED:.*]] = async.runtime.load %[[VAL_STORAGE]] : !async.value<f32>
// CHECK:   %[[RETURNED:.*]] = mulf %[[ARG]], %[[LOADED]] : f32
// CHECK:   async.runtime.store %[[RETURNED]], %[[RETURNED_STORAGE]] : !async.value<f32>
// CHECK:   async.runtime.set_available %[[RETURNED_STORAGE]]
// CHECK:   async.runtime.set_available %[[TOKEN]]
// CHECK:   br ^[[CLEANUP]]
  %3 = mulf %arg0, %2 : f32
  return %3: f32

// CHECK: ^[[BRANCH_ERROR]]:
// CHECK:   async.runtime.set_error %[[TOKEN]]
// CHECK:   async.runtime.set_error %[[RETURNED_STORAGE]]
// CHECK:   br ^[[CLEANUP]]


// CHECK: ^[[CLEANUP]]:
// CHECK:   async.coro.free %[[ID]], %[[HDL]]
// CHECK:   br ^[[SUSPEND]]

// CHECK: ^[[SUSPEND]]:
// CHECK:   async.coro.end %[[HDL]]
// CHECK:   return %[[TOKEN]], %[[RETURNED_STORAGE]] : !async.token, !async.value<f32>
}

// CHECK-LABEL: func @simple_caller()
// CHECK-SAME: -> (!async.token, !async.value<f32>)
func @simple_caller() -> f32 {
// CHECK: %[[TOKEN:.*]] = async.runtime.create : !async.token
// CHECK: %[[RETURNED_STORAGE:.*]] = async.runtime.create : !async.value<f32>
// CHECK: %[[ID:.*]] = async.coro.id
// CHECK: %[[HDL:.*]] = async.coro.begin %[[ID]]

// CHECK: %[[CONSTANT:.*]] = constant
  %c = constant 1.0 : f32
// CHECK: %[[RETURNED_TO_CALLER:.*]]:2 = call @simple_callee(%[[CONSTANT]]) : (f32) -> (!async.token, !async.value<f32>)
// CHECK: %[[SAVED:.*]] = async.coro.save %[[HDL]]
// CHECK: async.runtime.await_and_resume %[[RETURNED_TO_CALLER]]#0, %[[HDL]]
// CHECK: async.coro.suspend %[[SAVED]]
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME:.*]], ^[[CLEANUP:.*]]
  %r = call @simple_callee(%c): (f32) -> f32

// CHECK: ^[[RESUME]]:
// CHECK:   %[[IS_TOKEN_ERROR:.*]] = async.runtime.is_error %[[RETURNED_TO_CALLER]]#0 : !async.token
// CHECK:   cond_br %[[IS_TOKEN_ERROR]], ^[[BRANCH_ERROR:.*]], ^[[BRANCH_TOKEN_OK:.*]]

// CHECK: ^[[BRANCH_TOKEN_OK]]:
// CHECK:   %[[IS_VALUE_ERROR:.*]] = async.runtime.is_error %[[RETURNED_TO_CALLER]]#1 : !async.value<f32>
// CHECK:   cond_br %[[IS_VALUE_ERROR]], ^[[BRANCH_ERROR:.*]], ^[[BRANCH_VALUE_OK:.*]]

// CHECK: ^[[BRANCH_VALUE_OK]]:
// CHECK:   %[[LOADED:.*]] = async.runtime.load %[[RETURNED_TO_CALLER]]#1 : !async.value<f32>
// CHECK:   async.runtime.store %[[LOADED]], %[[RETURNED_STORAGE]] : !async.value<f32>
// CHECK:   async.runtime.set_available %[[RETURNED_STORAGE]]
// CHECK:   async.runtime.set_available %[[TOKEN]]
// CHECK:   br ^[[CLEANUP]]
  return %r: f32
// CHECK: ^[[BRANCH_ERROR]]:
// CHECK:   async.runtime.set_error %[[TOKEN]]
// CHECK:   async.runtime.set_error %[[RETURNED_STORAGE]]
// CHECK:   br ^[[CLEANUP]]


// CHECK: ^[[CLEANUP]]:
// CHECK:   async.coro.free %[[ID]], %[[HDL]]
// CHECK:   br ^[[SUSPEND]]

// CHECK: ^[[SUSPEND]]:
// CHECK:   async.coro.end %[[HDL]]
// CHECK:   return %[[TOKEN]], %[[RETURNED_STORAGE]] : !async.token, !async.value<f32>
}

// CHECK-LABEL: func @double_caller()
// CHECK-SAME: -> (!async.token, !async.value<f32>)
func @double_caller() -> f32 {
// CHECK: %[[TOKEN:.*]] = async.runtime.create : !async.token
// CHECK: %[[RETURNED_STORAGE:.*]] = async.runtime.create : !async.value<f32>
// CHECK: %[[ID:.*]] = async.coro.id
// CHECK: %[[HDL:.*]] = async.coro.begin %[[ID]]

// CHECK: %[[CONSTANT:.*]] = constant
  %c = constant 1.0 : f32
// CHECK: %[[RETURNED_TO_CALLER_1:.*]]:2 = call @simple_callee(%[[CONSTANT]]) : (f32) -> (!async.token, !async.value<f32>)
// CHECK: %[[SAVED_1:.*]] = async.coro.save %[[HDL]]
// CHECK: async.runtime.await_and_resume %[[RETURNED_TO_CALLER_1]]#0, %[[HDL]]
// CHECK: async.coro.suspend %[[SAVED_1]]
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME_1:.*]], ^[[CLEANUP:.*]]
  %r = call @simple_callee(%c): (f32) -> f32

// CHECK: ^[[RESUME_1]]:
// CHECK:   %[[IS_TOKEN_ERROR_1:.*]] = async.runtime.is_error %[[RETURNED_TO_CALLER_1]]#0 : !async.token
// CHECK:   cond_br %[[IS_TOKEN_ERROR_1]], ^[[BRANCH_ERROR:.*]], ^[[BRANCH_TOKEN_OK_1:.*]]

// CHECK: ^[[BRANCH_TOKEN_OK_1]]:
// CHECK:   %[[IS_VALUE_ERROR_1:.*]] = async.runtime.is_error %[[RETURNED_TO_CALLER_1]]#1 : !async.value<f32>
// CHECK:   cond_br %[[IS_VALUE_ERROR_1]], ^[[BRANCH_ERROR:.*]], ^[[BRANCH_VALUE_OK_1:.*]]

// CHECK: ^[[BRANCH_VALUE_OK_1]]:
// CHECK:   %[[LOADED_1:.*]] = async.runtime.load %[[RETURNED_TO_CALLER_1]]#1 : !async.value<f32>
// CHECK:   %[[RETURNED_TO_CALLER_2:.*]]:2 = call @simple_callee(%[[LOADED_1]]) : (f32) -> (!async.token, !async.value<f32>)
// CHECK:   %[[SAVED_2:.*]] = async.coro.save %[[HDL]]
// CHECK:   async.runtime.await_and_resume %[[RETURNED_TO_CALLER_2]]#0, %[[HDL]]
// CHECK:   async.coro.suspend %[[SAVED_2]]
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME_2:.*]], ^[[CLEANUP:.*]]
  %s = call @simple_callee(%r): (f32) -> f32

// CHECK: ^[[RESUME_2]]:
// CHECK:   %[[IS_TOKEN_ERROR_2:.*]] = async.runtime.is_error %[[RETURNED_TO_CALLER_2]]#0 : !async.token
// CHECK:   cond_br %[[IS_TOKEN_ERROR_2]], ^[[BRANCH_ERROR:.*]], ^[[BRANCH_TOKEN_OK_2:.*]]

// CHECK: ^[[BRANCH_TOKEN_OK_2]]:
// CHECK:   %[[IS_VALUE_ERROR_2:.*]] = async.runtime.is_error %[[RETURNED_TO_CALLER_2]]#1 : !async.value<f32>
// CHECK:   cond_br %[[IS_VALUE_ERROR_2]], ^[[BRANCH_ERROR:.*]], ^[[BRANCH_VALUE_OK_2:.*]]

// CHECK: ^[[BRANCH_VALUE_OK_2]]:
// CHECK:   %[[LOADED_2:.*]] = async.runtime.load %[[RETURNED_TO_CALLER_2]]#1 : !async.value<f32>
// CHECK:   async.runtime.store %[[LOADED_2]], %[[RETURNED_STORAGE]] : !async.value<f32>
// CHECK:   async.runtime.set_available %[[RETURNED_STORAGE]]
// CHECK:   async.runtime.set_available %[[TOKEN]]
// CHECK:   br ^[[CLEANUP]]
  return %s: f32
// CHECK: ^[[BRANCH_ERROR]]:
// CHECK:   async.runtime.set_error %[[TOKEN]]
// CHECK:   async.runtime.set_error %[[RETURNED_STORAGE]]
// CHECK:   br ^[[CLEANUP]]

// CHECK: ^[[CLEANUP]]:
// CHECK:   async.coro.free %[[ID]], %[[HDL]]
// CHECK:   br ^[[SUSPEND]]

// CHECK: ^[[SUSPEND]]:
// CHECK:   async.coro.end %[[HDL]]
// CHECK:   return %[[TOKEN]], %[[RETURNED_STORAGE]] : !async.token, !async.value<f32>
}

// CHECK-LABEL: func @recursive
// CHECK-SAME: (%[[ARG:.*]]: !async.token) -> !async.token
func @recursive(%arg: !async.token) {
// CHECK: %[[TOKEN:.*]] = async.runtime.create : !async.token
// CHECK: %[[ID:.*]] = async.coro.id
// CHECK: %[[HDL:.*]] = async.coro.begin %[[ID]]
// CHECK: %[[SAVED_1:.*]] = async.coro.save %[[HDL]]
// CHECK: async.runtime.await_and_resume %[[ARG]], %[[HDL]]
// CHECK: async.coro.suspend %[[SAVED_1]]
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME_1:.*]], ^[[CLEANUP:.*]]

  async.await %arg : !async.token
// CHECK: ^[[RESUME_1]]:
// CHECK:   %[[IS_ERROR:.*]] = async.runtime.is_error %[[ARG]] : !async.token
// CHECK:   cond_br %[[IS_ERROR]], ^[[BRANCH_ERROR:.*]], ^[[BRANCH_OK:.*]]

// CHECK: ^[[BRANCH_OK]]:
// CHECK:   %[[GIVEN:.*]] = async.runtime.create : !async.token
%r = async.runtime.create : !async.token
// CHECK:   async.runtime.set_available %[[GIVEN]]
async.runtime.set_available %r: !async.token
// CHECK:   %[[RETURNED_TO_CALLER:.*]] = call @recursive(%[[GIVEN]]) : (!async.token) -> !async.token
call @recursive(%r): (!async.token) -> ()
// CHECK: %[[SAVED_2:.*]] = async.coro.save %[[HDL]]
// CHECK: async.runtime.await_and_resume %[[RETURNED_TO_CALLER]], %[[HDL]]
// CHECK: async.coro.suspend %[[SAVED_2]]
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME_2:.*]], ^[[CLEANUP:.*]]

// CHECK: ^[[RESUME_2]]:
// CHECK:   async.runtime.set_available %[[TOKEN]]
// CHECK:   br ^[[CLEANUP]]

// CHECK: ^[[BRANCH_ERROR]]:
// CHECK:   async.runtime.set_error %[[TOKEN]]
// CHECK:   br ^[[CLEANUP]]
return

// CHECK: ^[[CLEANUP]]:
// CHECK:   async.coro.free %[[ID]], %[[HDL]]
// CHECK:   br ^[[SUSPEND]]

// CHECK: ^[[SUSPEND]]:
// CHECK:   async.coro.end %[[HDL]]
// CHECK:   return %[[TOKEN]] : !async.token
}

// CHECK-LABEL: func @corecursive1
// CHECK-SAME: (%[[ARG:.*]]: !async.token) -> !async.token
func @corecursive1(%arg: !async.token) {
// CHECK: %[[TOKEN:.*]] = async.runtime.create : !async.token
// CHECK: %[[ID:.*]] = async.coro.id
// CHECK: %[[HDL:.*]] = async.coro.begin %[[ID]]
// CHECK: %[[SAVED_1:.*]] = async.coro.save %[[HDL]]
// CHECK: async.runtime.await_and_resume %[[ARG]], %[[HDL]]
// CHECK: async.coro.suspend %[[SAVED_1]]
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME_1:.*]], ^[[CLEANUP:.*]]

  async.await %arg : !async.token
// CHECK: ^[[RESUME_1]]:
// CHECK:   %[[IS_ERROR:.*]] = async.runtime.is_error %[[ARG]] : !async.token
// CHECK:   cond_br %[[IS_ERROR]], ^[[BRANCH_ERROR:.*]], ^[[BRANCH_OK:.*]]

// CHECK: ^[[BRANCH_OK]]:
// CHECK:   %[[GIVEN:.*]] = async.runtime.create : !async.token
%r = async.runtime.create : !async.token
// CHECK:   async.runtime.set_available %[[GIVEN]]
async.runtime.set_available %r: !async.token
// CHECK:   %[[RETURNED_TO_CALLER:.*]] = call @corecursive2(%[[GIVEN]]) : (!async.token) -> !async.token
call @corecursive2(%r): (!async.token) -> ()
// CHECK: %[[SAVED_2:.*]] = async.coro.save %[[HDL]]
// CHECK: async.runtime.await_and_resume %[[RETURNED_TO_CALLER]], %[[HDL]]
// CHECK: async.coro.suspend %[[SAVED_2]]
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME_2:.*]], ^[[CLEANUP:.*]]

// CHECK: ^[[RESUME_2]]:
// CHECK:   async.runtime.set_available %[[TOKEN]]
// CHECK:   br ^[[CLEANUP]]

// CHECK: ^[[BRANCH_ERROR]]:
// CHECK:   async.runtime.set_error %[[TOKEN]]
// CHECK:   br ^[[CLEANUP]]
return

// CHECK: ^[[CLEANUP]]:
// CHECK:   async.coro.free %[[ID]], %[[HDL]]
// CHECK:   br ^[[SUSPEND]]

// CHECK: ^[[SUSPEND]]:
// CHECK:   async.coro.end %[[HDL]]
// CHECK:   return %[[TOKEN]] : !async.token
}

// CHECK-LABEL: func @corecursive2
// CHECK-SAME: (%[[ARG:.*]]: !async.token) -> !async.token
func @corecursive2(%arg: !async.token) {
// CHECK: %[[TOKEN:.*]] = async.runtime.create : !async.token
// CHECK: %[[ID:.*]] = async.coro.id
// CHECK: %[[HDL:.*]] = async.coro.begin %[[ID]]
// CHECK: %[[SAVED_1:.*]] = async.coro.save %[[HDL]]
// CHECK: async.runtime.await_and_resume %[[ARG]], %[[HDL]]
// CHECK: async.coro.suspend %[[SAVED_1]]
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME_1:.*]], ^[[CLEANUP:.*]]

  async.await %arg : !async.token
// CHECK: ^[[RESUME_1]]:
// CHECK:   %[[IS_ERROR:.*]] = async.runtime.is_error %[[ARG]] : !async.token
// CHECK:   cond_br %[[IS_ERROR]], ^[[BRANCH_ERROR:.*]], ^[[BRANCH_OK:.*]]

// CHECK: ^[[BRANCH_OK]]:
// CHECK:   %[[GIVEN:.*]] = async.runtime.create : !async.token
%r = async.runtime.create : !async.token
// CHECK:   async.runtime.set_available %[[GIVEN]]
async.runtime.set_available %r: !async.token
// CHECK:   %[[RETURNED_TO_CALLER:.*]] = call @corecursive1(%[[GIVEN]]) : (!async.token) -> !async.token
call @corecursive1(%r): (!async.token) -> ()
// CHECK: %[[SAVED_2:.*]] = async.coro.save %[[HDL]]
// CHECK: async.runtime.await_and_resume %[[RETURNED_TO_CALLER]], %[[HDL]]
// CHECK: async.coro.suspend %[[SAVED_2]]
// CHECK-SAME: ^[[SUSPEND:.*]], ^[[RESUME_2:.*]], ^[[CLEANUP:.*]]

// CHECK: ^[[RESUME_2]]:
// CHECK:   async.runtime.set_available %[[TOKEN]]
// CHECK:   br ^[[CLEANUP]]

// CHECK: ^[[BRANCH_ERROR]]:
// CHECK:   async.runtime.set_error %[[TOKEN]]
// CHECK:   br ^[[CLEANUP]]
return

// CHECK: ^[[CLEANUP]]:
// CHECK:   async.coro.free %[[ID]], %[[HDL]]
// CHECK:   br ^[[SUSPEND]]

// CHECK: ^[[SUSPEND]]:
// CHECK:   async.coro.end %[[HDL]]
// CHECK:   return %[[TOKEN]] : !async.token
}
