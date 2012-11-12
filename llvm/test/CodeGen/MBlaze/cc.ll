; Test some of the calling convention lowering done by the MBlaze backend.
; We test that integer values are passed in the correct registers and
; returned in the correct registers. Additionally, we test that the stack
; is used as appropriate for passing arguments that cannot be placed into
; registers.
;
; RUN: llc < %s -march=mblaze | FileCheck %s

declare i32 @printf(i8*, ...)
@MSG = internal constant [13 x i8] c"Message: %d\0A\00"

define void @params0_noret() {
    ; CHECK:        params0_noret:
    ret void
    ; CHECK-NOT:    {{.* r3, .*, .*}}
    ; CHECK-NOT:    {{.* r4, .*, .*}}
    ; CHECK:        rtsd
}

define i8 @params0_8bitret() {
    ; CHECK:        params0_8bitret:
    ret i8 1
    ; CHECK-NOT:    {{.* r3, .*, .*}}
    ; CHECK-NOT:    {{.* r4, .*, .*}}
    ; CHECK:        rtsd
    ; CHECK:        {{.* r3, r0, 1}}
}

define i16 @params0_16bitret() {
    ; CHECK:        params0_16bitret:
    ret i16 1
    ; CHECK:        rtsd
    ; CHECK:        {{.* r3, r0, 1}}
    ; CHECK-NOT:    {{.* r4, .*, .*}}
}

define i32 @params0_32bitret() {
    ; CHECK:        params0_32bitret:
    ret i32 1
    ; CHECK-NOT:    {{.* r4, .*, .*}}
    ; CHECK:        rtsd
    ; CHECK:        {{.* r3, r0, 1}}
}

define i64 @params0_64bitret() {
    ; CHECK:        params0_64bitret:
    ret i64 1
    ; CHECK:        {{.* r3, r0, .*}}
    ; CHECK:        rtsd
    ; CHECK:        {{.* r4, r0, 1}}
}

define i32 @params1_32bitret(i32 %a) {
    ; CHECK:        params1_32bitret:
    ret i32 %a
    ; CHECK-NOT:    {{.* r3, .*, .*}}
    ; CHECK-NOT:    {{.* r4, .*, .*}}
    ; CHECK:        rtsd
    ; CHECK:        {{.* r3, r5, r0}}
}

define i32 @params2_32bitret(i32 %a, i32 %b) {
    ; CHECK:        params2_32bitret:
    ret i32 %b
    ; CHECK-NOT:    {{.* r3, .*, .*}}
    ; CHECK-NOT:    {{.* r4, .*, .*}}
    ; CHECK:        rtsd
    ; CHECK:        {{.* r3, r6, r0}}
}

define i32 @params3_32bitret(i32 %a, i32 %b, i32 %c) {
    ; CHECK:        params3_32bitret:
    ret i32 %c
    ; CHECK-NOT:    {{.* r3, .*, .*}}
    ; CHECK-NOT:    {{.* r4, .*, .*}}
    ; CHECK:        rtsd
    ; CHECK:        {{.* r3, r7, r0}}
}

define i32 @params4_32bitret(i32 %a, i32 %b, i32 %c, i32 %d) {
    ; CHECK:        params4_32bitret:
    ret i32 %d
    ; CHECK-NOT:    {{.* r3, .*, .*}}
    ; CHECK-NOT:    {{.* r4, .*, .*}}
    ; CHECK:        rtsd
    ; CHECK:        {{.* r3, r8, r0}}
}

define i32 @params5_32bitret(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
    ; CHECK:        params5_32bitret:
    ret i32 %e
    ; CHECK-NOT:    {{.* r3, .*, .*}}
    ; CHECK-NOT:    {{.* r4, .*, .*}}
    ; CHECK:        rtsd
    ; CHECK:        {{.* r3, r9, r0}}
}

define i32 @params6_32bitret(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f) {
    ; CHECK:        params6_32bitret:
    ret i32 %f
    ; CHECK-NOT:    {{.* r3, .*, .*}}
    ; CHECK-NOT:    {{.* r4, .*, .*}}
    ; CHECK:        rtsd
    ; CHECK:        {{.* r3, r10, r0}}
}

define i32 @params7_32bitret(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f,
                             i32 %g) {
    ; CHECK:        params7_32bitret:
    ret i32 %g
    ; CHECK:        {{lwi? r3, r1, 32}}
    ; CHECK-NOT:    {{.* r4, .*, .*}}
    ; CHECK:        rtsd
}

define i32 @params8_32bitret(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f,
                             i32 %g, i32 %h) {
    ; CHECK:        params8_32bitret:
    ret i32 %h
    ; CHECK:        {{lwi? r3, r1, 36}}
    ; CHECK-NOT:    {{.* r4, .*, .*}}
    ; CHECK:        rtsd
}

define i32 @params9_32bitret(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f,
                             i32 %g, i32 %h, i32 %i) {
    ; CHECK:        params9_32bitret:
    ret i32 %i
    ; CHECK:        {{lwi? r3, r1, 40}}
    ; CHECK-NOT:    {{.* r4, .*, .*}}
    ; CHECK:        rtsd
}

define i32 @params10_32bitret(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f,
                              i32 %g, i32 %h, i32 %i, i32 %j) {
    ; CHECK:        params10_32bitret:
    ret i32 %j
    ; CHECK:        {{lwi? r3, r1, 44}}
    ; CHECK-NOT:    {{.* r4, .*, .*}}
    ; CHECK:        rtsd
}

define void @testing() {
    %MSG.1 = getelementptr [13 x i8]* @MSG, i32 0, i32 0

    call void @params0_noret()
    ; CHECK:        brlid

    %tmp.1 = call i8 @params0_8bitret()
    ; CHECK:        brlid
    call i32 (i8*,...)* @printf(i8* %MSG.1, i8 %tmp.1)

    %tmp.2 = call i16 @params0_16bitret()
    ; CHECK:        brlid
    call i32 (i8*,...)* @printf(i8* %MSG.1, i16 %tmp.2)

    %tmp.3 = call i32 @params0_32bitret()
    ; CHECK:        brlid
    call i32 (i8*,...)* @printf(i8* %MSG.1, i32 %tmp.3)

    %tmp.4 = call i64 @params0_64bitret()
    ; CHECK:        brlid
    call i32 (i8*,...)* @printf(i8* %MSG.1, i64 %tmp.4)

    %tmp.5 = call i32 @params1_32bitret(i32 1)
    ; CHECK:        {{.* r5, .*, .*}}
    ; CHECK:        brlid
    call i32 (i8*,...)* @printf(i8* %MSG.1, i32 %tmp.5)

    %tmp.6 = call i32 @params2_32bitret(i32 1, i32 2)
    ; CHECK:        {{.* r5, .*, .*}}
    ; CHECK:        {{.* r6, .*, .*}}
    ; CHECK:        brlid
    call i32 (i8*,...)* @printf(i8* %MSG.1, i32 %tmp.6)

    %tmp.7 = call i32 @params3_32bitret(i32 1, i32 2, i32 3)
    ; CHECK:        {{.* r5, .*, .*}}
    ; CHECK:        {{.* r6, .*, .*}}
    ; CHECK:        {{.* r7, .*, .*}}
    ; CHECK:        brlid
    call i32 (i8*,...)* @printf(i8* %MSG.1, i32 %tmp.7)

    %tmp.8 = call i32 @params4_32bitret(i32 1, i32 2, i32 3, i32 4)
    ; CHECK:        {{.* r5, .*, .*}}
    ; CHECK:        {{.* r6, .*, .*}}
    ; CHECK:        {{.* r7, .*, .*}}
    ; CHECK:        {{.* r8, .*, .*}}
    ; CHECK:        brlid
    call i32 (i8*,...)* @printf(i8* %MSG.1, i32 %tmp.8)

    %tmp.9 = call i32 @params5_32bitret(i32 1, i32 2, i32 3, i32 4, i32 5)
    ; CHECK:        {{.* r5, .*, .*}}
    ; CHECK:        {{.* r6, .*, .*}}
    ; CHECK:        {{.* r7, .*, .*}}
    ; CHECK:        {{.* r8, .*, .*}}
    ; CHECK:        {{.* r9, .*, .*}}
    ; CHECK:        brlid
    call i32 (i8*,...)* @printf(i8* %MSG.1, i32 %tmp.9)

    %tmp.10 = call i32 @params6_32bitret(i32 1, i32 2, i32 3, i32 4, i32 5,
                                         i32 6)
    ; CHECK:        {{.* r5, .*, .*}}
    ; CHECK:        {{.* r6, .*, .*}}
    ; CHECK:        {{.* r7, .*, .*}}
    ; CHECK:        {{.* r8, .*, .*}}
    ; CHECK:        {{.* r9, .*, .*}}
    ; CHECK:        {{.* r10, .*, .*}}
    ; CHECK:        brlid
    call i32 (i8*,...)* @printf(i8* %MSG.1, i32 %tmp.10)

    %tmp.11 = call i32 @params7_32bitret(i32 1, i32 2, i32 3, i32 4, i32 5,
                                         i32 6, i32 7)
    ; CHECK:        {{swi? .*, r1, 28}}
    ; CHECK:        {{.* r5, .*, .*}}
    ; CHECK:        {{.* r6, .*, .*}}
    ; CHECK:        {{.* r7, .*, .*}}
    ; CHECK:        {{.* r8, .*, .*}}
    ; CHECK:        {{.* r9, .*, .*}}
    ; CHECK:        {{.* r10, .*, .*}}
    ; CHECK:        brlid
    call i32 (i8*,...)* @printf(i8* %MSG.1, i32 %tmp.11)

    %tmp.12 = call i32 @params8_32bitret(i32 1, i32 2, i32 3, i32 4, i32 5,
                                         i32 6, i32 7, i32 8)
    ; CHECK:        {{swi? .*, r1, 32}}
    ; CHECK:        {{swi? .*, r1, 28}}
    ; CHECK:        {{.* r5, .*, .*}}
    ; CHECK:        {{.* r6, .*, .*}}
    ; CHECK:        {{.* r7, .*, .*}}
    ; CHECK:        {{.* r8, .*, .*}}
    ; CHECK:        {{.* r9, .*, .*}}
    ; CHECK:        {{.* r10, .*, .*}}
    ; CHECK:        brlid
    call i32 (i8*,...)* @printf(i8* %MSG.1, i32 %tmp.12)

    %tmp.13 = call i32 @params9_32bitret(i32 1, i32 2, i32 3, i32 4, i32 5,
                                         i32 6, i32 7, i32 8, i32 9)
    ; CHECK:        {{swi? .*, r1, 36}}
    ; CHECK:        {{swi? .*, r1, 32}}
    ; CHECK:        {{swi? .*, r1, 28}}
    ; CHECK:        {{.* r5, .*, .*}}
    ; CHECK:        {{.* r6, .*, .*}}
    ; CHECK:        {{.* r7, .*, .*}}
    ; CHECK:        {{.* r8, .*, .*}}
    ; CHECK:        {{.* r9, .*, .*}}
    ; CHECK:        {{.* r10, .*, .*}}
    ; CHECK:        brlid
    call i32 (i8*,...)* @printf(i8* %MSG.1, i32 %tmp.13)

    %tmp.14 = call i32 @params10_32bitret(i32 1, i32 2, i32 3, i32 4, i32 5,
                                          i32 6, i32 7, i32 8, i32 9, i32 10)
    ; CHECK:        {{swi? .*, r1, 40}}
    ; CHECK:        {{swi? .*, r1, 36}}
    ; CHECK:        {{swi? .*, r1, 32}}
    ; CHECK:        {{swi? .*, r1, 28}}
    ; CHECK:        {{.* r5, .*, .*}}
    ; CHECK:        {{.* r6, .*, .*}}
    ; CHECK:        {{.* r7, .*, .*}}
    ; CHECK:        {{.* r8, .*, .*}}
    ; CHECK:        {{.* r9, .*, .*}}
    ; CHECK:        {{.* r10, .*, .*}}
    ; CHECK:        brlid
    call i32 (i8*,...)* @printf(i8* %MSG.1, i32 %tmp.14)

    ret void
}
