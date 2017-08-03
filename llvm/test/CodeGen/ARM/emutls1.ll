; RUN: llc < %s -emulated-tls -march=arm -mtriple=arm-linux-androideabi \
; RUN:     | FileCheck %s
; RUN: llc < %s -emulated-tls -march=arm -mtriple=arm-linux-androideabi \
; RUN:     -relocation-model=pic | FileCheck %s --check-prefix=PIC

; Compared with tls1.ll, emulated mode should not use __aeabi_read_tp or __tls_get_addr.

; CHECK-NOT: _aeabi_read_tp
; CHECK-NOT: _tls_get_addr
; CHECK:     __emutls_get_addr
; CHECK-NOT: __aeabi_read_tp
; CHECK-NOT: _tls_get_addr

; PIC-NOT: _aeabi_read_tp
; PIC-NOT: _tls_get_addr
; PIC:     __emutls_get_addr
; PIC-NOT: _aeabi_read_tp
; PIC-NOT: _tls_get_addr

@i = thread_local global i32 15 ; <i32*> [#uses=2]

define i32 @f() {
entry:
 %tmp1 = load i32, i32* @i ; <i32> [#uses=1]
 ret i32 %tmp1
}

define i32* @g() {
entry:
 ret i32* @i
}
