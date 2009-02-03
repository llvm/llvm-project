; RUN: llvm-as <  %s | llc -mtriple=arm-apple-darwin | \
; RUN:   grep {\\.mod_init_func}
; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin | \
; RUN:   grep {\\.mod_term_func} 
; RUN: llvm-as  < %s | llc -mtriple=arm-linux-gnu | \
; RUN:   grep {\\.section \\.ctors,"aw",.progbits}
; RUN: llvm-as < %s | llc -mtriple=arm-linux-gnu | \
; RUN:   grep {\\.section \\.dtors,"aw",.progbits}
; RUN: llvm-as < %s | llc -mtriple=arm-linux-gnueabi | \
; RUN:   grep {\\.section \\.init_array,"aw",.init_array}
; RUN: llvm-as < %s | llc -mtriple=arm-linux-gnueabi | \
; RUN:   grep {\\.section \\.fini_array,"aw",.fini_array}

@llvm.global_ctors = appending global [1 x { i32, void ()* }] [ { i32, void ()* } { i32 65535, void ()* @__mf_init } ]                ; <[1 x { i32, void ()* }]*> [#uses=0]
@llvm.global_dtors = appending global [1 x { i32, void ()* }] [ { i32, void ()* } { i32 65535, void ()* @__mf_fini } ]                ; <[1 x { i32, void ()* }]*> [#uses=0]

define void @__mf_init() {
entry:
        ret void
}

define void @__mf_fini() {
entry:
        ret void
}
