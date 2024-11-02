; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: Function Attrs: memory(none)
; CHECK: @fn_readnone2()
declare void @fn_readnone2() memory(none)

; CHECK: Function Attrs: memory(read)
; CHECK: @fn_readonly()
declare void @fn_readonly() memory(read)

; CHECK: Function Attrs: memory(write)
; CHECK: @fn_writeonly()
declare void @fn_writeonly() memory(write)

; CHECK: Function Attrs: memory(readwrite)
; CHECK: @fn_readwrite()
declare void @fn_readwrite() memory(readwrite)

; CHECK: Function Attrs: memory(argmem: read)
; CHECK: @fn_argmem_read()
declare void @fn_argmem_read() memory(argmem: read)

; CHECK: Function Attrs: memory(argmem: write)
; CHECK: @fn_argmem_write()
declare void @fn_argmem_write() memory(argmem: write)

; CHECK: Function Attrs: memory(argmem: readwrite)
; CHECK: @fn_argmem_readwrite()
declare void @fn_argmem_readwrite() memory(argmem: readwrite)

; CHECK: Function Attrs: memory(inaccessiblemem: read)
; CHECK: @fn_inaccessiblemem_read()
declare void @fn_inaccessiblemem_read() memory(inaccessiblemem: read)

; CHECK: Function Attrs: memory(inaccessiblemem: write)
; CHECK: @fn_inaccessiblemem_write()
declare void @fn_inaccessiblemem_write() memory(inaccessiblemem: write)

; CHECK: Function Attrs: memory(inaccessiblemem: readwrite)
; CHECK: @fn_inaccessiblemem_readwrite()
declare void @fn_inaccessiblemem_readwrite() memory(inaccessiblemem: readwrite)

; CHECK: Function Attrs: memory(read, argmem: readwrite)
; CHECK: @fn_read_argmem_readwrite()
declare void @fn_read_argmem_readwrite() memory(read, argmem: readwrite)

; CHECK: Function Attrs: memory(read, argmem: write)
; CHECK: @fn_read_argmem_write()
declare void @fn_read_argmem_write() memory(read, argmem: write)

; CHECK: Function Attrs: memory(read, argmem: none)
; CHECK: @fn_read_argmem_none()
declare void @fn_read_argmem_none() memory(read, argmem: none)

; CHECK: Function Attrs: memory(argmem: read, inaccessiblemem: read)
; CHECK: @fn_argmem_inaccessiblemem_read()
declare void @fn_argmem_inaccessiblemem_read()
    memory(argmem: read, inaccessiblemem: read)

; CHECK: Function Attrs: memory(argmem: read, inaccessiblemem: write)
; CHECK: @fn_argmem_read_inaccessiblemem_write()
declare void @fn_argmem_read_inaccessiblemem_write()
    memory(argmem: read, inaccessiblemem: write)

; CHECK: Function Attrs: memory(argmem: read, inaccessiblemem: write)
; CHECK: @fn_argmem_read_inaccessiblemem_write_reordered()
declare void @fn_argmem_read_inaccessiblemem_write_reordered()
    memory(inaccessiblemem: write, argmem: read)
