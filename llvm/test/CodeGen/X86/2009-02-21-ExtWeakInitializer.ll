; RUN: llc < %s | FileCheck %s
; PR3629

; CHECK: weak
; CHECK-NEXT: weak
; CHECK-NEXT: weak
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "x86_64-unknown-freebsd7.1"
module asm ".ident\09\22$FreeBSD$\22"
	%struct.anon = type <{ %struct.uart_devinfo* }>
	%struct.lock_object = type <{ i8*, i32, i32, %struct.witness* }>
	%struct.mtx = type <{ %struct.lock_object, i64 }>
	%struct.uart_bas = type <{ i64, i64, i32, i32, i32, i8, i8, i8, i8 }>
	%struct.uart_class = type opaque
	%struct.uart_devinfo = type <{ %struct.anon, %struct.uart_ops*, %struct.uart_bas, i32, i32, i32, i32, i32, i8, i8, i8, i8, i32 (%struct.uart_softc*)*, i32 (%struct.uart_softc*)*, i8*, %struct.mtx* }>
	%struct.uart_ops = type <{ i32 (%struct.uart_bas*)*, void (%struct.uart_bas*, i32, i32, i32, i32)*, void (%struct.uart_bas*)*, void (%struct.uart_bas*, i32)*, i32 (%struct.uart_bas*)*, i32 (%struct.uart_bas*, %struct.mtx*)* }>
	%struct.uart_softc = type opaque
	%struct.witness = type opaque

@uart_classes = internal global [3 x %struct.uart_class*] [%struct.uart_class* @uart_ns8250_class, %struct.uart_class* @uart_sab82532_class, %struct.uart_class* @uart_z8530_class], align 8		; <[3 x %struct.uart_class*]*> [#uses=1]
@uart_ns8250_class = extern_weak global %struct.uart_class		; <%struct.uart_class*> [#uses=1]
@uart_sab82532_class = extern_weak global %struct.uart_class		; <%struct.uart_class*> [#uses=1]
@uart_z8530_class = extern_weak global %struct.uart_class		; <%struct.uart_class*> [#uses=1]
