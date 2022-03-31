; RUN: llc < %s -mtriple=arm-apple-darwin  | FileCheck %s -check-prefix=DARWIN
; RUN: llc < %s -mtriple=arm-apple-darwin -lower-global-dtors-via-cxa-atexit=false  | FileCheck %s -check-prefix=DARWIN-OLD
; RUN: llc < %s -mtriple=arm-linux-gnu -target-abi=apcs  | FileCheck %s -check-prefix=ELF
; RUN: llc < %s -mtriple=arm-linux-gnueabi | FileCheck %s -check-prefix=GNUEABI

; DARWIN: l_register_call_dtors:
; DARWIN: bl	___cxa_atexit
; DARWIN: .section	__DATA,__mod_init_func,mod_init_funcs
; DARWIN-NOT: __mod_term_func

; DARWIN-OLD: .section	__DATA,__mod_init_func,mod_init_funcs
; DARWIN-OLD: .section	__DATA,__mod_term_func,mod_term_funcs

; ELF: .section .ctors,"aw",%progbits
; ELF: .section .dtors,"aw",%progbits

; GNUEABI: .section .init_array,"aw",%init_array
; GNUEABI: .section .fini_array,"aw",%fini_array

@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [ { i32, void ()*, i8* } { i32 65535, void ()* @__mf_init, i8* null } ]                ; <[1 x { i32, void ()*, i8* }]*> [#uses=0]
@llvm.global_dtors = appending global [1 x { i32, void ()*, i8* }] [ { i32, void ()*, i8* } { i32 65535, void ()* @__mf_fini, i8* null } ]                ; <[1 x { i32, void ()*, i8* }]*> [#uses=0]

define void @__mf_init() {
entry:
        ret void
}

define void @__mf_fini() {
entry:
        ret void
}
