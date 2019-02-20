; RUN: llc -filetype=obj %s -thread-model=single -o - | obj2yaml | FileCheck %s --check-prefix=SINGLE
; RUN: llc -filetype=obj %s -thread-model=posix -o - | obj2yaml | FileCheck %s --check-prefix=THREADS

; Test that setting thread-model=posix causes data segments to be
; emitted as passive segments (i.e. have InitFlags set to 1).

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

@str = private unnamed_addr constant [7 x i8] c"Hello!\00", align 1

; SINGLE:       - Type:            DATA
; SINGLE-NEXT:    Segments:
; SINGLE-NEXT:      - SectionOffset:   6
; SINGLE-NEXT:        InitFlags:       0
; SINGLE-NEXT:        Offset:
; SINGLE-NEXT:          Opcode:          I32_CONST
; SINGLE-NEXT:          Value:           0
; SINGLE-NEXT:        Content:         48656C6C6F2100

; THREADS:       - Type:            DATA
; THREADS-NEXT:    Segments:
; THREADS-NEXT:      - SectionOffset:   3
; THREADS-NEXT:        InitFlags:       1
; THREADS-NEXT:        Content:         48656C6C6F2100
