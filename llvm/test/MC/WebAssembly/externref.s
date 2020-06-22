# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj < %s | obj2yaml | FileCheck %s

# Tests use of externref type, including storing it in a global

.globaltype my_global, externref

.globl read_externref
read_externref:
  .functype read_externref () -> (externref)
  global.get my_global
  end_function

.globl write_externref
write_externref:
  .functype write_externref (externref) -> ()
  local.get 0
  global.set my_global
  end_function

.globl call_with_ref
call_with_ref:
  .functype call_with_ref () -> ()
  call read_externref
  call write_externref
  end_function

my_global:

#      CHECK:  - Type:            GLOBAL
# CHECK-NEXT:    Globals:
# CHECK-NEXT:      - Index:           0
# CHECK-NEXT:        Type:            EXTERNREF
# CHECK-NEXT:        Mutable:         true
# CHECK-NEXT:        InitExpr:
# CHECK-NEXT:          Opcode:          REF_NULL
# CHECK-NEXT:          Type:            EXTERNREF
