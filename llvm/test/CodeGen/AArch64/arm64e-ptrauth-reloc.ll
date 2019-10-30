; RUN: llc < %s -mtriple arm64e-apple-darwin     -aarch64-ptrauth-global-dynamic-mat=1 | FileCheck %s --check-prefixes=CHECK,OPT,DYN-OPT
; RUN: llc < %s -mtriple arm64e-apple-darwin -O0 -aarch64-ptrauth-global-dynamic-mat=1 | FileCheck %s --check-prefixes=CHECK,O0,DYN-O0
; RUN: llc < %s -mtriple arm64e-apple-darwin     -aarch64-ptrauth-global-dynamic-mat=0 | FileCheck %s --check-prefixes=CHECK,OPT,LOAD-OPT
; RUN: llc < %s -mtriple arm64e-apple-darwin -O0 -aarch64-ptrauth-global-dynamic-mat=0 | FileCheck %s --check-prefixes=CHECK,O0,LOAD-O0

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; Check code references.

; LOAD-LABEL: _test_global_zero_disc:
; LOAD-NEXT: ; %bb.0:
; LOAD-NEXT: Lloh{{.*}}:
; LOAD-NEXT:  adrp x[[STUBPAGE:[0-9]+]], l_g$auth_ptr$ia$0@PAGE
; LOAD-NEXT: Lloh{{.*}}:
; LOAD-NEXT:  ldr x0, [x[[STUBPAGE]], l_g$auth_ptr$ia$0@PAGEOFF]
; LOAD-NEXT:  ret

; DYN-O0-LABEL: _test_global_zero_disc:
; DYN-O0-NEXT: ; %bb.0:
; DYN-O0-NEXT:  mov x8, #0
; DYN-O0-NEXT:   ; implicit-def: $x16
; DYN-O0-NEXT:   ; implicit-def: $x17
; DYN-O0-NEXT:  adrp x16, _g@GOTPAGE
; DYN-O0-NEXT:  ldr x16, [x16, _g@GOTPAGEOFF]
; DYN-O0-NEXT:  pacia x16, x8
; DYN-O0-NEXT:  mov x0, x16
; DYN-O0-NEXT:  ret

; DYN-OPT-LABEL: _test_global_zero_disc:
; DYN-OPT-NEXT: ; %bb.0:
; DYN-OPT-NEXT: Lloh{{.*}}:
; DYN-OPT-NEXT:  adrp x16, _g@GOTPAGE
; DYN-OPT-NEXT: Lloh{{.*}}:
; DYN-OPT-NEXT:  ldr x16, [x16, _g@GOTPAGEOFF]
; DYN-OPT-NEXT:  paciza x16
; DYN-OPT-NEXT:  mov x0, x16
; DYN-OPT-NEXT:  ret

define i8* @test_global_zero_disc() #0 {
  %tmp0 = bitcast { i8*, i32, i64, i64 }* @g.ptrauth.ia.0 to i8*
  ret i8* %tmp0
}

; LOAD-LABEL: _test_global_offset_zero_disc:
; LOAD-NEXT: ; %bb.0:
; LOAD-NEXT: Lloh{{.*}}:
; LOAD-NEXT:  adrp x[[STUBPAGE:[0-9]+]], l_g$148$auth_ptr$da$0@PAGE
; LOAD-NEXT: Lloh{{.*}}:
; LOAD-NEXT:  ldr x0, [x[[STUBPAGE]], l_g$148$auth_ptr$da$0@PAGEOFF]
; LOAD-NEXT:  ret

; DYN-O0-LABEL: _test_global_offset_zero_disc:
; DYN-O0-NEXT: ; %bb.0:
; DYN-O0-NEXT:  mov x8, #0
; DYN-O0-NEXT:   ; implicit-def: $x16
; DYN-O0-NEXT:   ; implicit-def: $x17
; DYN-O0-NEXT:  adrp x16, _g@GOTPAGE
; DYN-O0-NEXT:  ldr x16, [x16, _g@GOTPAGEOFF]
; DYN-O0-NEXT:  add x16, x16, #148
; DYN-O0-NEXT:  pacda x16, x8
; DYN-O0-NEXT:  mov x0, x16
; DYN-O0-NEXT:  ret

; DYN-OPT-LABEL: _test_global_offset_zero_disc:
; DYN-OPT-NEXT: ; %bb.0:
; DYN-OPT-NEXT: Lloh{{.*}}:
; DYN-OPT-NEXT:  adrp x16, _g@GOTPAGE
; DYN-OPT-NEXT: Lloh{{.*}}:
; DYN-OPT-NEXT:  ldr x16, [x16, _g@GOTPAGEOFF]
; DYN-OPT-NEXT:  add x16, x16, #148
; DYN-OPT-NEXT:  pacdza x16
; DYN-OPT-NEXT:  mov x0, x16
; DYN-OPT-NEXT:  ret

define i8* @test_global_offset_zero_disc() #0 {
  %tmp0 = bitcast { i8*, i32, i64, i64 }* @g.offset.ptrauth.da.0 to i8*
  ret i8* %tmp0
}

; For large offsets, materializing it can take up to 3 add instructions.
; We limit the offset to 32-bits.  We theoretically could support up to
; 64 bit offsets, but 32 bits Ought To Be Enough For Anybody, and that's
; the limit for the relocation addend anyway.

; LOAD-LABEL: _test_global_big_offset_zero_disc:
; LOAD-NEXT: ; %bb.0:
; LOAD-NEXT: Lloh{{.*}}:
; LOAD-NEXT:  adrp x[[STUBPAGE:[0-9]+]], l_g$2147549185$auth_ptr$da$0@PAGE
; LOAD-NEXT: Lloh{{.*}}:
; LOAD-NEXT:  ldr x0, [x[[STUBPAGE]], l_g$2147549185$auth_ptr$da$0@PAGEOFF]
; LOAD-NEXT:  ret

; DYN-O0-LABEL: _test_global_big_offset_zero_disc:
; DYN-O0-NEXT: ; %bb.0:
; DYN-O0-NEXT:  mov x8, #0
; DYN-O0-NEXT:   ; implicit-def: $x16
; DYN-O0-NEXT:   ; implicit-def: $x17
; DYN-O0-NEXT:  adrp x16, _g@GOTPAGE
; DYN-O0-NEXT:  ldr x16, [x16, _g@GOTPAGEOFF]
; DYN-O0-NEXT:  add x16, x16, #1
; DYN-O0-NEXT:  add x16, x16, #16, lsl #12
; DYN-O0-NEXT:  add x16, x16, #128, lsl #24
; DYN-O0-NEXT:  pacda x16, x8
; DYN-O0-NEXT:  mov x0, x16
; DYN-O0-NEXT:  ret

; DYN-OPT-LABEL: _test_global_big_offset_zero_disc:
; DYN-OPT-NEXT: ; %bb.0:
; DYN-OPT-NEXT: Lloh{{.*}}:
; DYN-OPT-NEXT:  adrp x16, _g@GOTPAGE
; DYN-OPT-NEXT: Lloh{{.*}}:
; DYN-OPT-NEXT:  ldr x16, [x16, _g@GOTPAGEOFF]
; DYN-OPT-NEXT:  add x16, x16, #1
; DYN-OPT-NEXT:  add x16, x16, #16, lsl #12
; DYN-OPT-NEXT:  add x16, x16, #128, lsl #24
; DYN-OPT-NEXT:  pacdza x16
; DYN-OPT-NEXT:  mov x0, x16
; DYN-OPT-NEXT:  ret

define i8* @test_global_big_offset_zero_disc() #0 {
  %tmp0 = bitcast { i8*, i32, i64, i64 }* @g.big_offset.ptrauth.da.0 to i8*
  ret i8* %tmp0
}

; LOAD-LABEL: _test_global_disc:
; LOAD-NEXT: ; %bb.0:
; LOAD-NEXT: Lloh{{.*}}:
; LOAD-NEXT:  adrp x[[STUBPAGE:[0-9]+]], l_g$auth_ptr$ia$42@PAGE
; LOAD-NEXT: Lloh{{.*}}:
; LOAD-NEXT:  ldr x0, [x[[STUBPAGE]], l_g$auth_ptr$ia$42@PAGEOFF]
; LOAD-NEXT:  ret

; DYN-O0-LABEL: _test_global_disc:
; DYN-O0-NEXT: ; %bb.0:
; DYN-O0-NEXT:  mov x8, #0
; DYN-O0-NEXT:   ; implicit-def: $x16
; DYN-O0-NEXT:   ; implicit-def: $x17
; DYN-O0-NEXT:  adrp x16, _g@GOTPAGE
; DYN-O0-NEXT:  ldr x16, [x16, _g@GOTPAGEOFF]
; DYN-O0-NEXT:  mov x17, #42
; DYN-O0-NEXT:  pacia x16, x17
; DYN-O0-NEXT:  mov x0, x16
; DYN-O0-NEXT:  ret

; DYN-OPT-LABEL: _test_global_disc:
; DYN-OPT-NEXT: ; %bb.0:
; DYN-OPT-NEXT: Lloh{{.*}}:
; DYN-OPT-NEXT:  adrp x16, _g@GOTPAGE
; DYN-OPT-NEXT: Lloh{{.*}}:
; DYN-OPT-NEXT:  ldr x16, [x16, _g@GOTPAGEOFF]
; DYN-OPT-NEXT:  mov x17, #42
; DYN-OPT-NEXT:  pacia x16, x17
; DYN-OPT-NEXT:  mov x0, x16
; DYN-OPT-NEXT:  ret

define i8* @test_global_disc() #0 {
  %tmp0 = bitcast { i8*, i32, i64, i64 }* @g.ptrauth.ia.42 to i8*
  ret i8* %tmp0
}

; LOAD-LABEL: _test_global_addr_disc:
; LOAD-NEXT: ; %bb.0:
; LOAD-NEXT: Lloh{{.*}}:
; LOAD-NEXT:  adrp x8, _g.ref.da.42.addr@PAGE
; LOAD-NEXT: Lloh{{.*}}:
; LOAD-NEXT:  add x8, x8, _g.ref.da.42.addr@PAGEOFF
; LOAD-NEXT:  movk x8, #42, lsl #48
; LOAD-NEXT: Lloh{{.*}}:
; LOAD-NEXT:  adrp x16, _g@GOTPAGE
; LOAD-NEXT: Lloh{{.*}}:
; LOAD-NEXT:  ldr x16, [x16, _g@GOTPAGEOFF]
; LOAD-NEXT:  pacda x16, x8
; LOAD-NEXT:  mov x0, x16
; LOAD-NEXT:  ret

; DYN-O0-LABEL: _test_global_addr_disc:
; DYN-O0-NEXT: ; %bb.0:
; DYN-O0-NEXT:  adrp x8, _g.ref.da.42.addr@PAGE
; DYN-O0-NEXT:  add x8, x8, _g.ref.da.42.addr@PAGEOFF
; DYN-O0-NEXT:  movk x8, #42, lsl #48
; DYN-O0-NEXT:   ; implicit-def: $x16
; DYN-O0-NEXT:   ; implicit-def: $x17
; DYN-O0-NEXT:  adrp x16, _g@GOTPAGE
; DYN-O0-NEXT:  ldr x16, [x16, _g@GOTPAGEOFF]
; DYN-O0-NEXT:  pacda x16, x8
; DYN-O0-NEXT:  mov x0, x16
; DYN-O0-NEXT:  ret

; DYN-OPT-LABEL: _test_global_addr_disc:
; DYN-OPT-NEXT: ; %bb.0:
; DYN-OPT-NEXT: Lloh{{.*}}:
; DYN-OPT-NEXT:  adrp x8, _g.ref.da.42.addr@PAGE
; DYN-OPT-NEXT: Lloh{{.*}}:
; DYN-OPT-NEXT:  add x8, x8, _g.ref.da.42.addr@PAGEOFF
; DYN-OPT-NEXT:  movk x8, #42, lsl #48
; DYN-OPT-NEXT: Lloh{{.*}}:
; DYN-OPT-NEXT:  adrp x16, _g@GOTPAGE
; DYN-OPT-NEXT: Lloh{{.*}}:
; DYN-OPT-NEXT:  ldr x16, [x16, _g@GOTPAGEOFF]
; DYN-OPT-NEXT:  pacda x16, x8
; DYN-OPT-NEXT:  mov x0, x16
; DYN-OPT-NEXT:  ret

define i8* @test_global_addr_disc() #0 {
  %tmp0 = bitcast { i8*, i32, i64, i64 }* @g.ptrauth.da.42.addr to i8*
  ret i8* %tmp0
}

; Process-specific keys can't use __DATA,__auth_ptr

; O0-LABEL: _test_global_process_specific:
; O0-NEXT: ; %bb.0:
; O0-NEXT:  mov x8, #0
; O0-NEXT:   ; implicit-def: $x16
; O0-NEXT:   ; implicit-def: $x17
; O0-NEXT:  adrp x16, _g@GOTPAGE
; O0-NEXT:  ldr x16, [x16, _g@GOTPAGEOFF]
; O0-NEXT:  pacib x16, x8
; O0-NEXT:  mov x0, x16
; O0-NEXT:  ret

; OPT-LABEL: _test_global_process_specific:
; OPT-NEXT: ; %bb.0:
; OPT-NEXT: Lloh{{.*}}:
; OPT-NEXT:  adrp x16, _g@GOTPAGE
; OPT-NEXT: Lloh{{.*}}:
; OPT-NEXT:  ldr x16, [x16, _g@GOTPAGEOFF]
; OPT-NEXT:  pacizb x16
; OPT-NEXT:  mov x0, x16
; OPT-NEXT:  ret

define i8* @test_global_process_specific() #0 {
  %tmp0 = bitcast { i8*, i32, i64, i64 }* @g.ptrauth.ib.0 to i8*
  ret i8* %tmp0
}

; weak symbols can't be assumed to be non-nil.  Use __DATA,__auth_ptr always.
; The alternative is to emit a null-check here, but that'd be redundant with
; whatever null-check follows in user code.

; O0-LABEL: _test_global_weak:
; O0-NEXT: ; %bb.0:
; O0-NEXT:  adrp x[[STUBPAGE:[0-9]+]], l_g_weak$auth_ptr$ia$42@PAGE
; O0-NEXT:  ldr x8, [x[[STUBPAGE]], l_g_weak$auth_ptr$ia$42@PAGEOFF]
; O0-NEXT:  mov x0, x8
; O0-NEXT:  ret

; OPT-LABEL: _test_global_weak:
; OPT-NEXT: ; %bb.0:
; OPT-NEXT: Lloh{{.*}}:
; OPT-NEXT:  adrp x[[STUBPAGE:[0-9]+]], l_g_weak$auth_ptr$ia$42@PAGE
; OPT-NEXT: Lloh{{.*}}:
; OPT-NEXT:  ldr x0, [x[[STUBPAGE]], l_g_weak$auth_ptr$ia$42@PAGEOFF]
; OPT-NEXT:  ret

define i8* @test_global_weak() #0 {
  %tmp0 = bitcast { i8*, i32, i64, i64 }* @g_weak.ptrauth.ia.42 to i8*
  ret i8* %tmp0
}

attributes #0 = { nounwind }

; Check global references.

@g = external global i32

@g_weak = extern_weak global i32

; CHECK-LABEL:   .section __DATA,__const
; CHECK-NEXT:    .globl _g.ref.ia.0
; CHECK-NEXT:    .p2align 4
; CHECK-NEXT:  _g.ref.ia.0:
; CHECK-NEXT:    .quad 5
; CHECK-NEXT:    .quad _g@AUTH(ia,0)
; CHECK-NEXT:    .quad 6

@g.ptrauth.ia.0 = private constant { i8*, i32, i64, i64 } { i8* bitcast (i32* @g to i8*), i32 0, i64 0, i64 0 }, section "llvm.ptrauth"

@g.ref.ia.0 = constant { i64, i8*, i64 } { i64 5, i8* bitcast ({ i8*, i32, i64, i64 }* @g.ptrauth.ia.0 to i8*), i64 6 }

; CHECK-LABEL:   .globl _g.ref.ia.42
; CHECK-NEXT:    .p2align 3
; CHECK-NEXT:  _g.ref.ia.42:
; CHECK-NEXT:    .quad _g@AUTH(ia,42)

@g.ptrauth.ia.42 = private constant { i8*, i32, i64, i64 } { i8* bitcast (i32* @g to i8*), i32 0, i64 0, i64 42 }, section "llvm.ptrauth"

@g.ref.ia.42 = constant i8* bitcast ({ i8*, i32, i64, i64 }* @g.ptrauth.ia.42 to i8*)

; CHECK-LABEL:   .globl _g.ref.ib.0
; CHECK-NEXT:    .p2align 4
; CHECK-NEXT:  _g.ref.ib.0:
; CHECK-NEXT:    .quad 5
; CHECK-NEXT:    .quad _g@AUTH(ib,0)
; CHECK-NEXT:    .quad 6

@g.ptrauth.ib.0 = private constant { i8*, i32, i64, i64 } { i8* bitcast (i32* @g to i8*), i32 1, i64 0, i64 0 }, section "llvm.ptrauth"

@g.ref.ib.0 = constant { i64, i8*, i64 } { i64 5, i8* bitcast ({ i8*, i32, i64, i64 }* @g.ptrauth.ib.0 to i8*), i64 6 }


; CHECK-LABEL:   .globl _g.ref.da.42.addr
; CHECK-NEXT:    .p2align 3
; CHECK-NEXT:  _g.ref.da.42.addr:
; CHECK-NEXT:    .quad _g@AUTH(da,42,addr)

@g.ptrauth.da.42.addr = private constant { i8*, i32, i64, i64 } { i8* bitcast (i32* @g to i8*), i32 2, i64 ptrtoint (i8** @g.ref.da.42.addr to i64), i64 42 }, section "llvm.ptrauth"

@g.ref.da.42.addr = constant i8* bitcast ({ i8*, i32, i64, i64 }* @g.ptrauth.da.42.addr to i8*)

; CHECK-LABEL:   .globl _g.offset.ref.da.0
; CHECK-NEXT:    .p2align 3
; CHECK-NEXT:  _g.offset.ref.da.0:
; CHECK-NEXT:    .quad (_g+148)@AUTH(da,0)

@g.offset.ptrauth.da.0 = private constant { i8*, i32, i64, i64 } { i8* getelementptr inbounds (i8, i8* bitcast (i32* @g to i8*), i64 148), i32 2, i64 0, i64 0 }, section "llvm.ptrauth"

@g.offset.ref.da.0 = constant i8* bitcast ({ i8*, i32, i64, i64 }* @g.offset.ptrauth.da.0 to i8*)

; CHECK-LABEL:   .globl _g.big_offset.ref.da.0
; CHECK-NEXT:    .p2align 3
; CHECK-NEXT:  _g.big_offset.ref.da.0:
; CHECK-NEXT:    .quad (_g+2147549185)@AUTH(da,0)

@g.big_offset.ptrauth.da.0 = private constant { i8*, i32, i64, i64 } { i8* getelementptr inbounds (i8, i8* bitcast (i32* @g to i8*), i64 add (i64 2147483648, i64 65537)), i32 2, i64 0, i64 0 }, section "llvm.ptrauth"

@g.big_offset.ref.da.0 = constant i8* bitcast ({ i8*, i32, i64, i64 }* @g.big_offset.ptrauth.da.0 to i8*)

; CHECK-LABEL:   .globl _g.weird_ref.da.0
; CHECK-NEXT:    .p2align 3
; CHECK-NEXT:  _g.weird_ref.da.0:
; CHECK-NEXT:    .quad (_g+148)@AUTH(da,0)

@g.weird_ref.da.0 = constant i64 ptrtoint (i8* bitcast (i64* inttoptr (i64 ptrtoint (i8* bitcast ({ i8*, i32, i64, i64 }* @g.offset.ptrauth.da.0 to i8*) to i64) to i64*) to i8*) to i64)

; CHECK-LABEL:   .globl _g_weak.ref.ia.42
; CHECK-NEXT:    .p2align 3
; CHECK-NEXT:  _g_weak.ref.ia.42:
; CHECK-NEXT:    .quad _g_weak@AUTH(ia,42)

@g_weak.ptrauth.ia.42 = private constant { i8*, i32, i64, i64 } { i8* bitcast (i32* @g_weak to i8*), i32 0, i64 0, i64 42 }, section "llvm.ptrauth"

@g_weak.ref.ia.42 = constant i8* bitcast ({ i8*, i32, i64, i64 }* @g_weak.ptrauth.ia.42 to i8*)
