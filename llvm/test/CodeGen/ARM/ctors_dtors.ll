; RUN: llc < %s -mtriple=arm-apple-darwin  | FileCheck %s -check-prefix=DARWIN
; RUN: llc < %s -mtriple=arm-apple-darwin -disable-atexit-based-global-dtor-lowering  | FileCheck %s -check-prefix=DARWIN-LEGACY
; RUN: llc < %s -mtriple=arm-linux-gnu -target-abi=apcs  | FileCheck %s -check-prefix=ELF
; RUN: llc < %s -mtriple=arm-linux-gnueabi | FileCheck %s -check-prefix=GNUEABI

; DARWIN: l_register_call_dtors:
; DARWIN: bl	___cxa_atexit
; DARWIN: .section	__DATA,__mod_init_func,mod_init_funcs
; DARWIN-NOT: __mod_term_func

; DARWIN-LEGACY-NOT: atexit
; DARWIN-LEGACY: .section	__DATA,__mod_init_func,mod_init_funcs
; DARWIN-LEGACY: .section	__DATA,__mod_term_func,mod_term_funcs

; ELF: .section .ctors,"aw",%progbits
; ELF: .section .dtors,"aw",%progbits

; GNUEABI: .section .init_array,"aw",%init_array
; GNUEABI: .section .fini_array,"aw",%fini_array

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [ { i32, ptr, ptr } { i32 65535, ptr @__mf_init, ptr null } ]                ; <ptr> [#uses=0]
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [ { i32, ptr, ptr } { i32 65535, ptr @__mf_fini, ptr null } ]                ; <ptr> [#uses=0]

define void @__mf_init() {
entry:
        ret void
}

define void @__mf_fini() {
entry:
        ret void
}
