; RUN: llc < %s -relocation-model=static | FileCheck %s -check-prefix=STATIC
; RUN: llc < %s -relocation-model=pic | FileCheck %s -check-prefix=PIC

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

@a = internal unnamed_addr constant [2 x i32] [i32 1, i32 2]
@a1 = unnamed_addr constant [2 x i32] [i32 1, i32 2]
@e = internal  unnamed_addr constant [2 x [2 x i32]] [[2 x i32] [i32 1, i32 2], [2 x i32] [i32 3, i32 4]], align 16
@e1 = unnamed_addr constant [2 x [2 x i32]] [[2 x i32] [i32 1, i32 2], [2 x i32] [i32 3, i32 4]], align 16
@p = unnamed_addr constant i8* bitcast ([2 x i32]* @a to i8*)
@t = unnamed_addr constant i8* bitcast ([2 x [2 x i32]]* @e to i8*)
@p1 = unnamed_addr constant i8* bitcast ([2 x i32]* @a1 to i8*)
@t1 = unnamed_addr constant i8* bitcast ([2 x [2 x i32]]* @e1 to i8*)
@p2 = internal global i8* bitcast([2 x i32]* @a1 to i8*)
@t2 = internal global i8* bitcast([2 x [2 x i32]]* @e1 to i8*)
@p3 = internal global i8* bitcast([2 x i32]* @a to i8*)
@t3 = internal global i8* bitcast([2 x [2 x i32]]* @e to i8*)

; STATIC: .section .rodata.cst8,"aM",@progbits,8
; STATIC: a:
; STATIC: a1:
; STATIC: .section .rodata.cst16,"aM",@progbits,16
; STATIC: e:
; STATIC: e1:
; STATIC: .section .rodata,"a",@progbits
; STATIC: p:

; PIC: .section .rodata.cst8,"aM",@progbits,8
; PIC: a:
; PIC: a1:
; PIC: .section .rodata.cst16,"aM",@progbits,16
; PIC: e:
; PIC: e1:
; PIC: .section .data.rel.ro,"aw",@progbits
; PIC: p:
; PIC: t:
; PIC-NOT: .section
; PIC: p1:
; PIC: t1:
; PIC: .data
; PIC: p2:
; PIC: t2:
; PIC-NOT: .section
; PIC: p3:
; PIC: t3:
