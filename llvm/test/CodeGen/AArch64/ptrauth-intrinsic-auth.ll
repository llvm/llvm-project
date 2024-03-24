; RUN: llc -mtriple=aarch64 -verify-machineinstrs -asm-verbose=0              -stop-after=finalize-isel < %s \
; RUN:     | FileCheck --check-prefixes=MIR-HINT %s
; RUN: llc -mtriple=aarch64 -verify-machineinstrs -asm-verbose=0 -mattr=v8.3a -stop-after=finalize-isel < %s \
; RUN:     | FileCheck --check-prefixes=MIR-V83 %s
; RUN: llc -mtriple=aarch64 -verify-machineinstrs -asm-verbose=0              -stop-after=finalize-isel -global-isel=1 -global-isel-abort=1 < %s \
; RUN:     | FileCheck --check-prefixes=MIR-HINT %s
; RUN: llc -mtriple=aarch64 -verify-machineinstrs -asm-verbose=0 -mattr=v8.3a -stop-after=finalize-isel -global-isel=1 -global-isel-abort=1 < %s \
; RUN:     | FileCheck --check-prefixes=MIR-V83 %s

; RUN: llc -mtriple=aarch64 -verify-machineinstrs -asm-verbose=0              < %s \
; RUN:     | FileCheck --check-prefixes=HINT-DEFAULT %s
; RUN: llc -mtriple=aarch64 -verify-machineinstrs -asm-verbose=0 -mattr=v8.3a < %s \
; RUN:     | FileCheck --check-prefixes=V83,V83-DEFAULT %s
; RUN: llc -mtriple=aarch64 -verify-machineinstrs -asm-verbose=0 -mattr=v8.3a -aarch64-intrinsic-check-method=none            < %s \
; RUN:     | FileCheck --check-prefixes=V83,V83-NONE %s
; RUN: llc -mtriple=aarch64 -verify-machineinstrs -asm-verbose=0 -mattr=v8.3a -aarch64-intrinsic-check-method=load            < %s \
; RUN:     | FileCheck --check-prefixes=V83,V83-LDR %s
; RUN: llc -mtriple=aarch64 -verify-machineinstrs -asm-verbose=0 -mattr=v8.3a -aarch64-intrinsic-check-method=high-bits-notbi < %s \
; RUN:     | FileCheck --check-prefixes=V83,V83-BITS-NOTBI %s
; RUN: llc -mtriple=aarch64 -verify-machineinstrs -asm-verbose=0 -mattr=v8.3a -aarch64-intrinsic-check-method=xpac            < %s \
; RUN:     | FileCheck --check-prefixes=V83,V83-XPAC %s

; Check that the expected instruction sequences are emitted between
; authentication and return from function.
define i64 @test_check_methods(i64 %signed) {
; V83-NONE-LABEL:  test_check_methods:
; V83-NONE-NEXT:     .cfi_startproc
; V83-NONE-NEXT:     autiza x0
; V83-NONE-NEXT:     ret

; V83-LDR-LABEL:  test_check_methods:
; V83-LDR-NEXT:     .cfi_startproc
; V83-LDR-NEXT:     autiza x0
; V83-LDR-NEXT:     ldr w8, [x0]
; V83-LDR-NEXT:     ret

; V83-BITS-NOTBI-LABEL:  test_check_methods:
; V83-BITS-NOTBI-NEXT:     .cfi_startproc
; V83-BITS-NOTBI-NEXT:     autiza  x0
; V83-BITS-NOTBI-NEXT:     eor x8, x0, x0, lsl #1
; V83-BITS-NOTBI-NEXT:     tbnz x8, #62, .[[FAIL:LBB[_0-9]+]]
; V83-BITS-NOTBI-NEXT:     ret
; V83-BITS-NOTBI-NEXT:   .[[FAIL]]:
; V83-BITS-NOTBI-NEXT:     brk #0xc470

; V83-XPAC-LABEL:  test_check_methods:
; V83-XPAC-NEXT:     .cfi_startproc
; V83-XPAC-NEXT:     autiza x0
; V83-XPAC-NEXT:     mov x8, x0
; V83-XPAC-NEXT:     xpaci x8
; V83-XPAC-NEXT:     cmp x8, x0
; V83-XPAC-NEXT:     b.ne .[[FAIL:LBB[_0-9]+]]
; V83-XPAC-NEXT:     ret
; V83-XPAC-NEXT:   .[[FAIL]]:
; V83-XPAC-NEXT:     brk #0xc470
  %auted = call i64 @llvm.ptrauth.auth(i64 %signed, i32 0, i64 0)
  ret i64 %auted
}

; Check for correct XPAC(I|D) instruction and operand of BRK instruction,
; once per key.

define i64 @test_xpac_and_brk_ia(i64 %signed) {
; V83-XPAC-LABEL:  test_xpac_and_brk_ia:
; V83-XPAC-NEXT:     .cfi_startproc
; V83-XPAC-NEXT:     autiza x0
; V83-XPAC-NEXT:     mov x8, x0
; V83-XPAC-NEXT:     xpaci x8
; V83-XPAC-NEXT:     cmp x8, x0
; V83-XPAC-NEXT:     b.ne .[[FAIL:LBB[_0-9]+]]
; V83-XPAC-NEXT:     ret
; V83-XPAC-NEXT:   .[[FAIL]]:
; V83-XPAC-NEXT:     brk #0xc470
  %auted = call i64 @llvm.ptrauth.auth(i64 %signed, i32 0, i64 0)
  ret i64 %auted
}

define i64 @test_xpac_and_brk_ib(i64 %signed) {
; V83-XPAC-LABEL:  test_xpac_and_brk_ib:
; V83-XPAC-NEXT:     .cfi_startproc
; V83-XPAC-NEXT:     autizb x0
; V83-XPAC-NEXT:     mov x8, x0
; V83-XPAC-NEXT:     xpaci x8
; V83-XPAC-NEXT:     cmp x8, x0
; V83-XPAC-NEXT:     b.ne .[[FAIL:LBB[_0-9]+]]
; V83-XPAC-NEXT:     ret
; V83-XPAC-NEXT:   .[[FAIL]]:
; V83-XPAC-NEXT:     brk #0xc471
  %auted = call i64 @llvm.ptrauth.auth(i64 %signed, i32 1, i64 0)
  ret i64 %auted
}

define i64 @test_xpac_and_brk_da(i64 %signed) "target-features"="+v8.3a" {
; V83-XPAC-LABEL:  test_xpac_and_brk_da:
; V83-XPAC-NEXT:     .cfi_startproc
; V83-XPAC-NEXT:     autdza x0
; V83-XPAC-NEXT:     mov x8, x0
; V83-XPAC-NEXT:     xpacd x8
; V83-XPAC-NEXT:     cmp x8, x0
; V83-XPAC-NEXT:     b.ne .[[FAIL:LBB[_0-9]+]]
; V83-XPAC-NEXT:     ret
; V83-XPAC-NEXT:   .[[FAIL]]:
; V83-XPAC-NEXT:     brk #0xc472
  %auted = call i64 @llvm.ptrauth.auth(i64 %signed, i32 2, i64 0)
  ret i64 %auted
}

define i64 @test_xpac_and_brk_db(i64 %signed) "target-features"="+v8.3a" {
; V83-XPAC-LABEL:  test_xpac_and_brk_db:
; V83-XPAC-NEXT:     .cfi_startproc
; V83-XPAC-NEXT:     autdzb x0
; V83-XPAC-NEXT:     mov x8, x0
; V83-XPAC-NEXT:     xpacd x8
; V83-XPAC-NEXT:     cmp x8, x0
; V83-XPAC-NEXT:     b.ne .[[FAIL:LBB[_0-9]+]]
; V83-XPAC-NEXT:     ret
; V83-XPAC-NEXT:   .[[FAIL]]:
; V83-XPAC-NEXT:     brk #0xc473
  %auted = call i64 @llvm.ptrauth.auth(i64 %signed, i32 3, i64 0)
  ret i64 %auted
}

; Check for correct AUT* instruction (raw discriminator)
; and default check method.

define i64 @test_aut_ia(i64 %signed, i64 %disc) {
; HINT-DEFAULT-LABEL:  test_aut_ia:
; HINT-DEFAULT-NEXT:     .cfi_startproc
; HINT-DEFAULT-NEXT:     mov x17, x0
; HINT-DEFAULT-NEXT:     mov x16, x1
; HINT-DEFAULT-NEXT:     hint #12
; HINT-DEFAULT-NEXT:     mov x0, x17
; HINT-DEFAULT-NEXT:     ret

; V83-DEFAULT-LABEL:  test_aut_ia:
; V83-DEFAULT-NEXT:     .cfi_startproc
; V83-DEFAULT-NEXT:     autia x0, x1
; V83-DEFAULT-NEXT:     mov x1, x0
; V83-DEFAULT-NEXT:     xpaci x1
; V83-DEFAULT-NEXT:     cmp x1, x0
; V83-DEFAULT-NEXT:     b.ne .[[FAIL:LBB[_0-9]+]]
; V83-DEFAULT-NEXT:     ret
; V83-DEFAULT-NEXT:   .[[FAIL]]:
; V83-DEFAULT-NEXT:     brk #0xc470
  %auted = call i64 @llvm.ptrauth.auth(i64 %signed, i32 0, i64 %disc)
  ret i64 %auted
}

define i64 @test_aut_ib(i64 %signed, i64 %disc) {
; HINT-DEFAULT-LABEL:  test_aut_ib:
; HINT-DEFAULT-NEXT:     .cfi_startproc
; HINT-DEFAULT-NEXT:     mov x17, x0
; HINT-DEFAULT-NEXT:     mov x16, x1
; HINT-DEFAULT-NEXT:     hint #14
; HINT-DEFAULT-NEXT:     mov x0, x17
; HINT-DEFAULT-NEXT:     ret

; V83-DEFAULT-LABEL:  test_aut_ib:
; V83-DEFAULT-NEXT:     .cfi_startproc
; V83-DEFAULT-NEXT:     autib x0, x1
; V83-DEFAULT-NEXT:     mov x1, x0
; V83-DEFAULT-NEXT:     xpaci x1
; V83-DEFAULT-NEXT:     cmp x1, x0
; V83-DEFAULT-NEXT:     b.ne .[[FAIL:LBB[_0-9]+]]
; V83-DEFAULT-NEXT:     ret
; V83-DEFAULT-NEXT:   .[[FAIL]]:
; V83-DEFAULT-NEXT:     brk #0xc471
  %auted = call i64 @llvm.ptrauth.auth(i64 %signed, i32 1, i64 %disc)
  ret i64 %auted
}

define i64 @test_aut_da(i64 %signed, i64 %disc) "target-features"="+v8.3a" {
; V83-DEFAULT-LABEL:  test_aut_da:
; V83-DEFAULT-NEXT:     .cfi_startproc
; V83-DEFAULT-NEXT:     autda x0, x1
; V83-DEFAULT-NEXT:     ldr w1, [x0]
; V83-DEFAULT-NEXT:     ret
  %auted = call i64 @llvm.ptrauth.auth(i64 %signed, i32 2, i64 %disc)
  ret i64 %auted
}

define i64 @test_aut_db(i64 %signed, i64 %disc) "target-features"="+v8.3a" {
; V83-DEFAULT-LABEL:  test_aut_db:
; V83-DEFAULT-NEXT:     .cfi_startproc
; V83-DEFAULT-NEXT:     autdb x0, x1
; V83-DEFAULT-NEXT:     ldr w1, [x0]
; V83-DEFAULT-NEXT:     ret
  %auted = call i64 @llvm.ptrauth.auth(i64 %signed, i32 3, i64 %disc)
  ret i64 %auted
}

; Check for correct AUT* instruction (zero discriminator).

define i64 @test_aut_ia_zero(i64 %signed) {
; HINT-DEFAULT-LABEL:  test_aut_ia_zero:
; HINT-DEFAULT-NEXT:     .cfi_startproc
; HINT-DEFAULT-NEXT:     mov x17, x0
; HINT-DEFAULT-NEXT:     mov x16, #0
; HINT-DEFAULT-NEXT:     hint #12

; V83-LABEL:  test_aut_ia_zero:
; V83-NEXT:     .cfi_startproc
; V83-NEXT:     autiza x0
  %auted = call i64 @llvm.ptrauth.auth(i64 %signed, i32 0, i64 0)
  ret i64 %auted
}

define i64 @test_aut_ib_zero(i64 %signed) {
; HINT-DEFAULT-LABEL:  test_aut_ib_zero:
; HINT-DEFAULT-NEXT:     .cfi_startproc
; HINT-DEFAULT-NEXT:     mov x17, x0
; HINT-DEFAULT-NEXT:     mov x16, #0
; HINT-DEFAULT-NEXT:     hint #14

; V83-LABEL:  test_aut_ib_zero:
; V83-NEXT:     .cfi_startproc
; V83-NEXT:     autizb x0
  %auted = call i64 @llvm.ptrauth.auth(i64 %signed, i32 1, i64 0)
  ret i64 %auted
}

define i64 @test_aut_da_zero(i64 %signed) "target-features"="+v8.3a" {
; V83-LABEL:  test_aut_da_zero:
; V83-NEXT:     .cfi_startproc
; V83-NEXT:     autdza x0
  %auted = call i64 @llvm.ptrauth.auth(i64 %signed, i32 2, i64 0)
  ret i64 %auted
}

define i64 @test_aut_db_zero(i64 %signed) "target-features"="+v8.3a" {
; V83-LABEL:  test_aut_db_zero:
; V83-NEXT:     .cfi_startproc
; V83-NEXT:     autdzb x0
  %auted = call i64 @llvm.ptrauth.auth(i64 %signed, i32 3, i64 0)
  ret i64 %auted
}

; Check different signing schemas, both assembler output and MIR instructions.
;
; IB key is used instead of IA just because it is not encoded as zero
; by AArch64 backend.

define i64 @test_zero_disc(i64 %signed) {
; HINT-DEFAULT-LABEL:  test_zero_disc:
; HINT-DEFAULT-NEXT:     .cfi_startproc
; HINT-DEFAULT-NEXT:     mov     x17, x0
; HINT-DEFAULT-NEXT:     mov     x16, #0
; HINT-DEFAULT-NEXT:     hint    #14

; V83-LABEL:  test_zero_disc:
; V83-NEXT:     .cfi_startproc
; V83-NEXT:     autizb x0

; MIR-HINT-LABEL:  name: test_zero_disc
; MIR-HINT:        body:
; MIR-HINT:          bb{{.*}}:
; MIR-HINT:            liveins: $x0
; MIR-HINT-DAG:        %[[SIGNED:[0-9]+]]:gpr64common = COPY $x0
; MIR-HINT:            $x17, {{(dead )?}}$xzr = PAUTH_AUTH %[[SIGNED]], $xzr, 0, 0, 1, implicit-def $x16{{$}}
; MIR-HINT:            %[[AUTED:[0-9]+]]:gpr64common = COPY $x17
; MIR-HINT:            $x0 = COPY %[[AUTED]]
; MIR-HINT:            RET_ReallyLR implicit $x0

; MIR-V83-LABEL:  name: test_zero_disc
; MIR-V83:        body:
; MIR-V83:          bb{{.*}}:
; MIR-V83:            liveins: $x0
; MIR-V83-DAG:        %[[SIGNED:[0-9]+]]:gpr64common = COPY $x0
; MIR-V83:            %[[AUTED:[0-9]+]]:gpr64common, {{(dead )?}}$xzr = PAUTH_AUTH %[[SIGNED]], $xzr, 0, 0, 1, implicit-def %{{[0-9]+}}{{$}}
; MIR-V83:            $x0 = COPY %[[AUTED]]
; MIR-V83:            RET_ReallyLR implicit $x0
  %auted = call i64 @llvm.ptrauth.auth(i64 %signed, i32 1, i64 0)
  ret i64 %auted
}

define i64 @test_immediate_disc(i64 %signed) {
; HINT-DEFAULT-LABEL:  test_immediate_disc:
; HINT-DEFAULT-NEXT:     .cfi_startproc
; HINT-DEFAULT-NEXT:     mov     x17, x0
; HINT-DEFAULT-NEXT:     mov     x16, #42
; HINT-DEFAULT-NEXT:     hint    #14

; V83-LABEL:  test_immediate_disc:
; V83-NEXT:     .cfi_startproc
; V83-NEXT:     mov x8, #42
; V83-NEXT:     autib x0, x8

; MIR-HINT-LABEL:  name: test_immediate_disc
; MIR-HINT:        body:
; MIR-HINT:          bb{{.*}}:
; MIR-HINT:            liveins: $x0
; MIR-HINT-DAG:        %[[SIGNED:[0-9]+]]:gpr64common = COPY $x0
; MIR-HINT:            $x17, {{(dead )?}}$xzr = PAUTH_AUTH %[[SIGNED]], $xzr, 42, 0, 1, implicit-def $x16{{$}}
; MIR-HINT:            %[[AUTED:[0-9]+]]:gpr64common = COPY $x17
; MIR-HINT:            $x0 = COPY %[[AUTED]]
; MIR-HINT:            RET_ReallyLR implicit $x0

; MIR-V83-LABEL:  name: test_immediate_disc
; MIR-V83:        body:
; MIR-V83:          bb{{.*}}:
; MIR-V83:            liveins: $x0
; MIR-V83-DAG:        %[[SIGNED:[0-9]+]]:gpr64common = COPY $x0
; MIR-V83:            %[[AUTED:[0-9]+]]:gpr64common, {{(dead )?}}$xzr = PAUTH_AUTH %[[SIGNED]], $xzr, 42, 0, 1, implicit-def %{{[0-9]+}}{{$}}
; MIR-V83:            $x0 = COPY %[[AUTED]]
; MIR-V83:            RET_ReallyLR implicit $x0
  %auted = call i64 @llvm.ptrauth.auth(i64 %signed, i32 1, i64 42)
  ret i64 %auted
}

define i64 @test_raw_disc(i64 %signed, i64 %raw_disc) {
; HINT-DEFAULT-LABEL:  test_raw_disc:
; HINT-DEFAULT-NEXT:     .cfi_startproc
; HINT-DEFAULT-NEXT:     mov     x17, x0
; HINT-DEFAULT-NEXT:     mov     x16, x1
; HINT-DEFAULT-NEXT:     hint    #14

; V83-LABEL:  test_raw_disc:
; V83-NEXT:     .cfi_startproc
; V83-NEXT:     autib x0, x1

; MIR-HINT-LABEL:  name: test_raw_disc
; MIR-HINT:        body:
; MIR-HINT:          bb{{.*}}:
; MIR-HINT:            liveins: $x0, $x1
; MIR-HINT-DAG:        %[[SIGNED:[0-9]+]]:gpr64common = COPY $x0
; MIR-HINT-DAG:        %[[RAW_DISC:[0-9]+]]:gpr64 = COPY $x1
; MIR-HINT:            $x17, {{(dead )?}}$x16 = PAUTH_AUTH %[[SIGNED]], %[[RAW_DISC]], 0, 0, 1{{$}}
; MIR-HINT:            %[[AUTED:[0-9]+]]:gpr64common = COPY $x17
; MIR-HINT:            $x0 = COPY %[[AUTED]]
; MIR-HINT:            RET_ReallyLR implicit $x0

; MIR-V83-LABEL:  name: test_raw_disc
; MIR-V83:        body:
; MIR-V83:          bb{{.*}}:
; MIR-V83:            liveins: $x0, $x1
; MIR-V83-DAG:        %[[SIGNED:[0-9]+]]:gpr64common = COPY $x0
; MIR-V83-DAG:        %[[RAW_DISC:[0-9]+]]:gpr64 = COPY $x1
; MIR-V83:            %[[AUTED:[0-9]+]]:gpr64common, {{(dead )?}}%{{[0-9]+}}:gpr64 = PAUTH_AUTH %[[SIGNED]], %[[RAW_DISC]], 0, 0, 1{{$}}
; MIR-V83:            $x0 = COPY %[[AUTED]]
; MIR-V83:            RET_ReallyLR implicit $x0
  %auted = call i64 @llvm.ptrauth.auth(i64 %signed, i32 1, i64 %raw_disc)
  ret i64 %auted
}

define i64 @test_blended_disc(i64 %signed, i64 %addr_disc) {
; HINT-DEFAULT-LABEL:  test_blended_disc:
; HINT-DEFAULT-NEXT:     .cfi_startproc
; HINT-DEFAULT-NEXT:     mov     x17, x0
; HINT-DEFAULT-NEXT:     mov     x16, x1
; HINT-DEFAULT-NEXT:     movk    x16, #42, lsl #48
; HINT-DEFAULT-NEXT:     hint    #14

; V83-LABEL:  test_blended_disc:
; V83-NEXT:     .cfi_startproc
; V83-NEXT:     movk x1, #42, lsl #48
; V83-NEXT:     autib x0, x1

; MIR-HINT-LABEL:  name: test_blended_disc
; MIR-HINT:        body:
; MIR-HINT:          bb{{.*}}:
; MIR-HINT:            liveins: $x0, $x1
; MIR-HINT-DAG:        %[[SIGNED:[0-9]+]]:gpr64common = COPY $x0
; MIR-HINT-DAG:        %[[ADDR_DISC:[0-9]+]]:gpr64 = COPY $x1
; MIR-HINT:            $x17, {{(dead )?}}$x16 = PAUTH_AUTH %[[SIGNED]], %[[ADDR_DISC]], 42, 1, 1{{$}}
; MIR-HINT:            %[[AUTED:[0-9]+]]:gpr64common = COPY $x17
; MIR-HINT:            $x0 = COPY %[[AUTED]]
; MIR-HINT:            RET_ReallyLR implicit $x0

; MIR-V83-LABEL:  name: test_blended_disc
; MIR-V83:        body:
; MIR-V83:          bb{{.*}}:
; MIR-V83:            liveins: $x0, $x1
; MIR-V83-DAG:        %[[SIGNED:[0-9]+]]:gpr64common = COPY $x0
; MIR-V83-DAG:        %[[ADDR_DISC:[0-9]+]]:gpr64 = COPY $x1
; MIR-V83:            %[[AUTED:[0-9]+]]:gpr64common, {{(dead )?}}%{{[0-9]+}}:gpr64 = PAUTH_AUTH %[[SIGNED]], %[[ADDR_DISC]], 42, 1, 1{{$}}
; MIR-V83:            $x0 = COPY %[[AUTED]]
; MIR-V83:            RET_ReallyLR implicit $x0
  %disc = call i64 @llvm.ptrauth.blend(i64 %addr_disc, i64 42)
  %auted = call i64 @llvm.ptrauth.auth(i64 %signed, i32 1, i64 %disc)
  ret i64 %auted
}

define i64 @test_blended_with_zero(i64 %signed, i64 %addr_disc) {
; HINT-DEFAULT-LABEL:  test_blended_with_zero:
; HINT-DEFAULT-NEXT:     .cfi_startproc
; HINT-DEFAULT-NEXT:     mov     x17, x0
; HINT-DEFAULT-NEXT:     mov     x16, x1
; HINT-DEFAULT-NEXT:     movk    x16, #0, lsl #48
; HINT-DEFAULT-NEXT:     hint    #14

; V83-LABEL:  test_blended_with_zero:
; V83-NEXT:     .cfi_startproc
; V83-NEXT:     movk x1, #0, lsl #48
; V83-NEXT:     autib x0, x1

; MIR-HINT-LABEL:  name: test_blended_with_zero
; MIR-HINT:        body:
; MIR-HINT:          bb{{.*}}:
; MIR-HINT:            liveins: $x0, $x1
; MIR-HINT-DAG:        %[[SIGNED:[0-9]+]]:gpr64common = COPY $x0
; MIR-HINT-DAG:        %[[ADDR_DISC:[0-9]+]]:gpr64 = COPY $x1
; MIR-HINT:            $x17, {{(dead )?}}$x16 = PAUTH_AUTH %[[SIGNED]], %[[ADDR_DISC]], 0, 1, 1{{$}}
; MIR-HINT:            %[[AUTED:[0-9]+]]:gpr64common = COPY $x17
; MIR-HINT:            $x0 = COPY %[[AUTED]]
; MIR-HINT:            RET_ReallyLR implicit $x0

; MIR-V83-LABEL:  name: test_blended_with_zero
; MIR-V83:        body:
; MIR-V83:          bb{{.*}}:
; MIR-V83:            liveins: $x0, $x1
; MIR-V83-DAG:        %[[SIGNED:[0-9]+]]:gpr64common = COPY $x0
; MIR-V83-DAG:        %[[ADDR_DISC:[0-9]+]]:gpr64 = COPY $x1
; MIR-V83:            %[[AUTED:[0-9]+]]:gpr64common, {{(dead )?}}%{{[0-9]+}}:gpr64 = PAUTH_AUTH %[[SIGNED]], %[[ADDR_DISC]], 0, 1, 1{{$}}
; MIR-V83:            $x0 = COPY %[[AUTED]]
; MIR-V83:            RET_ReallyLR implicit $x0
  %disc = call i64 @llvm.ptrauth.blend(i64 %addr_disc, i64 0)
  %auted = call i64 @llvm.ptrauth.auth(i64 %signed, i32 1, i64 %disc)
  ret i64 %auted
}

define i64 @test_null_blend(i64 %signed) {
; HINT-DEFAULT-LABEL:  test_null_blend:
; HINT-DEFAULT-NEXT:     .cfi_startproc
; HINT-DEFAULT-NEXT:     mov     x17, x0
; HINT-DEFAULT-NEXT:     mov     x16, xzr
; HINT-DEFAULT-NEXT:     movk    x16, #42, lsl #48
; HINT-DEFAULT-NEXT:     hint    #14

; V83-LABEL:  test_null_blend:
; V83-NEXT:     .cfi_startproc
; V83-NEXT:     mov x8, xzr
; V83-NEXT:     movk x8, #42, lsl #48
; V83-NEXT:     autib x0, x8

; MIR-HINT-LABEL:  name: test_null_blend
; MIR-HINT:        body:
; MIR-HINT:          bb{{.*}}:
; MIR-HINT:            liveins: $x0
; MIR-HINT-DAG:        %[[SIGNED:[0-9]+]]:gpr64common = COPY $x0
; MIR-HINT-DAG:        %[[ZERO:[0-9]+]]:gpr64 = COPY $xzr
; MIR-HINT:            $x17, {{(dead )?}}$x16 = PAUTH_AUTH %[[SIGNED]], %[[ZERO]], 42, 1, 1{{$}}
; MIR-HINT:            %[[AUTED:[0-9]+]]:gpr64common = COPY $x17
; MIR-HINT:            $x0 = COPY %[[AUTED]]
; MIR-HINT:            RET_ReallyLR implicit $x0

; MIR-V83-LABEL:  name: test_null_blend
; MIR-V83:        body:
; MIR-V83:          bb{{.*}}:
; MIR-V83:            liveins: $x0
; MIR-V83-DAG:        %[[SIGNED:[0-9]+]]:gpr64common = COPY $x0
; MIR-V83-DAG:        %[[ZERO:[0-9]+]]:gpr64 = COPY $xzr
; MIR-V83:            %[[AUTED:[0-9]+]]:gpr64common, {{(dead )?}}%{{[0-9]+}}:gpr64 = PAUTH_AUTH %[[SIGNED]], %[[ZERO]], 42, 1, 1{{$}}
; MIR-V83:            $x0 = COPY %[[AUTED]]
; MIR-V83:            RET_ReallyLR implicit $x0
  %disc = call i64 @llvm.ptrauth.blend(i64 0, i64 42)
  %auted = call i64 @llvm.ptrauth.auth(i64 %signed, i32 1, i64 %disc)
  ret i64 %auted
}

define i64 @test_custom_discriminator(i64 %signed, i64 %n) {
; HINT-DEFAULT-LABEL:  test_custom_discriminator:
; HINT-DEFAULT-NEXT:     .cfi_startproc
; HINT-DEFAULT-NEXT:     add x16, x1, #42
; HINT-DEFAULT-NEXT:     mov x17, x0
; HINT-DEFAULT-NEXT:     hint #14

; V83-LABEL:  test_custom_discriminator:
; V83-NEXT:     .cfi_startproc
; V83-NEXT:     add     x8, x1, #42
; V83-NEXT:     autib   x0, x8

; MIR-HINT-LABEL:  name: test_custom_discriminator
; MIR-HINT:        body:
; MIR-HINT:          bb{{.*}}:
; MIR-HINT:            liveins: $x0, $x1
; MIR-HINT-DAG:        %[[SIGNED:[0-9]+]]:gpr64common = COPY $x0
; MIR-HINT-DAG:        %[[N:[0-9]+]]:gpr64{{.*}} = COPY $x1
; MIR-HINT-DAG:        %[[DISC:[0-9]+]]:gpr64common = ADDXri %[[N]], 42, 0
; MIR-HINT:            $x17, {{(dead )?}}$x16 = PAUTH_AUTH %[[SIGNED]], %[[DISC]], 0, 0, 1{{$}}
; MIR-HINT:            %[[AUTED:[0-9]+]]:gpr64common = COPY $x17
; MIR-HINT:            $x0 = COPY %[[AUTED]]
; MIR-HINT:            RET_ReallyLR implicit $x0

; MIR-V83-LABEL:  name: test_custom_discriminator
; MIR-V83:        body:
; MIR-V83:          bb{{.*}}:
; MIR-V83:            liveins: $x0, $x1
; MIR-V83-DAG:        %[[SIGNED:[0-9]+]]:gpr64common = COPY $x0
; MIR-V83-DAG:        %[[N:[0-9]+]]:gpr64{{.*}} = COPY $x1
; MIR-V83-DAG:        %[[DISC:[0-9]+]]:gpr64common = ADDXri %[[N]], 42, 0
; MIR-V83:            %[[AUTED:[0-9]+]]:gpr64common, {{(dead )?}}%{{[0-9]+}}:gpr64 = PAUTH_AUTH %[[SIGNED]], %[[DISC]], 0, 0, 1{{$}}
; MIR-V83:            $x0 = COPY %[[AUTED]]
; MIR-V83:            RET_ReallyLR implicit $x0
  %disc = add i64 %n, 42
  %auted = call i64 @llvm.ptrauth.auth(i64 %signed, i32 1, i64 %disc)
  ret i64 %auted
}

declare ptr @llvm.frameaddress(i32)
declare i64 @llvm.ptrauth.auth(i64, i32, i64)
declare i64 @llvm.ptrauth.blend(i64, i64)
