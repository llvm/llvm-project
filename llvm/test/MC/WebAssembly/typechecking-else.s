# Check that it converts to .o without errors
# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj -mattr=+reference-types < %s

objects:
    .tabletype objects,externref

handle_value:
    .hidden handle_value
    .globl handle_value
    .type handle_value,@function
    .functype       handle_value (i32) -> (externref)
    local.get 0
    i32.const -1
    i32.eq
    if externref
    ref.null_extern
    else
    local.get 0
    table.get objects
    end_if
    end_function
