; RUN: llc -march=mipsel -relocation-model=pic -disable-mips-delay-filler < %s | FileCheck %s

@foo = thread_local global i32 42
@bar = hidden alias i32* @foo

define i32* @zed() {
; CHECK: __tls_get_addr
; CHECK-NEXT: %tlsgd(bar)
       ret i32* @bar
}
