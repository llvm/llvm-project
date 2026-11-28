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

; CHECK: Function Attrs: memory(errnomem: read)
; CHECK: @fn_errnomem_read()
declare void @fn_errnomem_read() memory(errnomem: read)

; CHECK: Function Attrs: memory(errnomem: write)
; CHECK: @fn_errnomem_write()
declare void @fn_errnomem_write() memory(errnomem: write)

; CHECK: Function Attrs: memory(errnomem: readwrite)
; CHECK: @fn_errnomem_readwrite()
declare void @fn_errnomem_readwrite() memory(errnomem: readwrite)

; CHECK: Function Attrs: memory(threadid: read)
; CHECK: @fn_threadid_read()
declare void @fn_threadid_read() memory(threadid: read)

; CHECK: Function Attrs: memory(threadid: write)
; CHECK: @fn_threadid_write()
declare void @fn_threadid_write() memory(threadid: write)

; CHECK: Function Attrs: memory(threadid: readwrite)
; CHECK: @fn_threadid_readwrite()
declare void @fn_threadid_readwrite() memory(threadid: readwrite)

; The default access kind applies to threadid as well.
; CHECK: Function Attrs: memory(read)
; CHECK: @fn_read_threadid_read()
declare void @fn_read_threadid_read() memory(read, threadid: read)

; CHECK: Function Attrs: memory(readwrite, threadid: read)
; CHECK: @fn_readwrite_threadid_read()
declare void @fn_readwrite_threadid_read() memory(readwrite, threadid: read)

; CHECK: Function Attrs: memory(read, inaccessiblemem: none, threadid: none)
; CHECK: @fn_read_inaccessiblemem_none_threadid_none()
declare void @fn_read_inaccessiblemem_none_threadid_none()
    memory(read, threadid: none, inaccessiblemem: none)

; CHECK: Function Attrs: memory(argmem: read, threadid: write, target_mem: read)
; CHECK: @fn_argmem_read_threadid_write_target_mem_read()
declare void @fn_argmem_read_threadid_write_target_mem_read()
    memory(target_mem: read, threadid: write, argmem: read)

; CHECK: Function Attrs: memory(write, threadid: read, target_mem0: none)
; CHECK: @fn_write_threadid_read_target_mem0_none()
declare void @fn_write_threadid_read_target_mem0_none()
    memory(write, threadid: read, target_mem0: none)

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

; CHECK: Function Attrs: memory(target_mem0: write)
; CHECK: @fn_write_mem_target0()
declare void @fn_write_mem_target0()
    memory(target_mem0: write)

; CHECK: Function Attrs: memory(target_mem0: read)
; CHECK: @fn_read_mem_target0()
declare void @fn_read_mem_target0()
    memory(target_mem0: read)

; CHECK: Function Attrs: memory(target_mem1: write)
; CHECK: @fn_write_target_mem1()
declare void @fn_write_target_mem1()
    memory(target_mem1: write)

; CHECK: Function Attrs: memory(target_mem1: read)
; CHECK: @fn_read_target_mem1()
declare void @fn_read_target_mem1()
    memory(target_mem1: read)

; CHECK: Function Attrs: memory(target_mem0: read, target_mem1: write)
; CHECK: @fn_read_target_mem0_write_mem_target1()
declare void @fn_read_target_mem0_write_mem_target1()
    memory(target_mem0: read, target_mem1: write)

; CHECK: Function Attrs: memory(inaccessiblemem: write)
; CHECK: @fn_inaccessiblemem_write_new()
declare void @fn_inaccessiblemem_write_new()
    memory(inaccessiblemem: write)

; CHECK: Function Attrs: memory(inaccessiblemem: read, target_mem: read)
; CHECK: @fn_inaccessiblemem_target_mem0_1read()
declare void @fn_inaccessiblemem_target_mem0_1read()
    memory(inaccessiblemem: read, target_mem0: read, target_mem1: read)

; CHECK: Function Attrs: memory(target_mem0: read)
; CHECK: @fn_inaccessiblemem_none_target_mem0_read()
declare void @fn_inaccessiblemem_none_target_mem0_read()
    memory(inaccessiblemem: none, target_mem0: read)

; CHECK: Function Attrs: memory(write, inaccessiblemem: read)
; CHECK: @fn_write_inaccessiblemem_read_target_mem0_write
declare void @fn_write_inaccessiblemem_read_target_mem0_write()
    memory(write, inaccessiblemem: read, target_mem0: write)

; CHECK: Function Attrs: memory(write, target_mem0: read)
; CHECK: @fn_write_inaccessiblemem_write_target_mem0_read()
declare void @fn_write_inaccessiblemem_write_target_mem0_read()
    memory(write, inaccessiblemem: write, target_mem0: read)

; CHECK: Function Attrs: memory(write, target_mem0: read)
; CHECK: @fn_write_target_mem0_readwrite()
declare void @fn_write_target_mem0_readwrite()
    memory(write, target_mem0: read)

; CHECK: Function Attrs: memory(target_mem: read)
; CHECK: @fn_write_target_mem
declare void @fn_write_target_mem()
    memory(target_mem: read)

; CHECK: Function Attrs: memory(target_mem0: write, target_mem1: read)
; CHECK: @fn_write_target_mem_read_write
declare void @fn_write_target_mem_read_write()
    memory(target_mem: read, target_mem0:write)

; CHECK: Function Attrs: memory(readwrite, target_mem: read)
; CHECK: @fn_all_readwrite
declare void @fn_all_readwrite()
    memory(readwrite, target_mem: read)
