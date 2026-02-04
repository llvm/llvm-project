; RUN: llvm-as < %s | llvm-dis | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; CHECK: Function Attrs: memory(target_mem1: /*aarch64_za:*/ write)
; CHECK: @fn_inaccessiblemem_write_aarch64_za() [[ATTR0:#.*]]
declare void @fn_inaccessiblemem_write_aarch64_za()
    memory(target_mem1: /*aarch64_za:*/ write)

; CHECK: Function Attrs: memory(target_mem1: /*aarch64_za:*/ read)
; CHECK: @fn_inaccessiblemem_read_aarch64_za() [[ATTR1:#.*]]
declare void @fn_inaccessiblemem_read_aarch64_za()
    memory(target_mem1: /*aarch64_za:*/ read)

; CHECK: Function Attrs: memory(target_mem0: /*aarch64_fpmr*/ write)
; CHECK: @fn_inaccessiblemem_write_aarch64_fpmr() [[ATTR2:#.*]]
declare void @fn_inaccessiblemem_write_aarch64_fpmr()
    memory(target_mem0: /*aarch64_fpmr*/ write)

; CHECK: ; Function Attrs: memory(target_mem0: /*aarch64_fpmr*/ read)
; CHECK: @fn_inaccessiblemem_read_aarch64_fpmr()  [[ATTR3:#.*]]
declare void @fn_inaccessiblemem_read_aarch64_fpmr()
    memory(target_mem0: /*aarch64_fpmr*/ read)

; CHECK: Function Attrs: memory(target_mem0: /*aarch64_fpmr*/ read, target_mem1: /*aarch64_za:*/ write)
; CHECK: @fn_inaccessiblemem_read_aarch64_fpmr_write_aarch64_za() [[ATTR4:#.*]]
declare void @fn_inaccessiblemem_read_aarch64_fpmr_write_aarch64_za()
    memory(target_mem0: /*aarch64_fpmr*/ read, target_mem1: /*aarch64_za:*/ write)

; CHECK: attributes [[ATTR0]] = { memory(target_mem1: /*aarch64_za:*/ write) }
; CHECK: attributes [[ATTR1]] = { memory(target_mem1: /*aarch64_za:*/ read) }
; CHECK: attributes [[ATTR2]] = { memory(target_mem0: /*aarch64_fpmr*/ write) }
; CHECK: attributes [[ATTR3]] = { memory(target_mem0: /*aarch64_fpmr*/ read) }
; CHECK: attributes [[ATTR4]] = { memory(target_mem0: /*aarch64_fpmr*/ read, target_mem1: /*aarch64_za:*/ write) }
