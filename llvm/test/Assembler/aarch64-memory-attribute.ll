; RUN: llvm-as < %s | llvm-dis | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; CHECK: Function Attrs: memory(target_mem1: write)
; CHECK: @fn_inaccessiblemem_write_target_mem1() [[ATTR0:#.*]]
declare void @fn_inaccessiblemem_write_target_mem1()
    memory(target_mem1: write)

; CHECK: Function Attrs: memory(target_mem1: read)
; CHECK: @fn_inaccessiblemem_read_target_mem1() [[ATTR1:#.*]]
declare void @fn_inaccessiblemem_read_target_mem1()
    memory(target_mem1: read)

; CHECK: Function Attrs: memory(target_mem0: write)
; CHECK: @fn_inaccessiblemem_write_target_mem0() [[ATTR2:#.*]]
declare void @fn_inaccessiblemem_write_target_mem0()
    memory(target_mem0: write)

; CHECK: ; Function Attrs: memory(target_mem0: read)
; CHECK: @fn_inaccessiblemem_read_target_mem0()  [[ATTR3:#.*]]
declare void @fn_inaccessiblemem_read_target_mem0()
    memory(target_mem0: read)

; CHECK: Function Attrs: memory(target_mem0: read, target_mem1: write)
; CHECK: @fn_inaccessiblemem_read_target_mem0_write_target_mem1() [[ATTR4:#.*]]
declare void @fn_inaccessiblemem_read_target_mem0_write_target_mem1()
    memory(target_mem0: read, target_mem1: write)

; CHECK: attributes [[ATTR0]] = { memory(target_mem1: write) }
; CHECK: attributes [[ATTR1]] = { memory(target_mem1: read) }
; CHECK: attributes [[ATTR2]] = { memory(target_mem0: write) }
; CHECK: attributes [[ATTR3]] = { memory(target_mem0: read) }
; CHECK: attributes [[ATTR4]] = { memory(target_mem0: read, target_mem1: write) }
