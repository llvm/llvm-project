; RUN: llc < %s | FileCheck %s

target triple = "i686-pc-linux-gnu"
@str = internal constant [9 x i8] c"%f+%f*i\0A\00"              ; <ptr> [#uses=1]

define i32 @main() {
; CHECK-LABEL: main:
; CHECK-NOT: ret
; CHECK: subl $12, %esp
; CHECK: pushl
; CHECK: pushl
; CHECK: pushl
; CHECK: pushl
; CHECK: pushl
; CHECK: calll cexp
; CHECK: addl $28, %esp
; CHECK: ret

entry:
        %retval = alloca i32, align 4           ; <ptr> [#uses=1]
        %tmp = alloca { double, double }, align 16              ; <ptr> [#uses=4]
        %tmp1 = alloca { double, double }, align 16             ; <ptr> [#uses=4]
        %tmp2 = alloca { double, double }, align 16             ; <ptr> [#uses=3]
        %pi = alloca double, align 8            ; <ptr> [#uses=2]
        %z = alloca { double, double }, align 16                ; <ptr> [#uses=4]
        %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
        store double 0x400921FB54442D18, ptr %pi
        %tmp.upgrd.1 = load double, ptr %pi         ; <double> [#uses=1]
        %real = getelementptr { double, double }, ptr %tmp1, i64 0, i32 0           ; <ptr> [#uses=1]
        store double 0.000000e+00, ptr %real
        %real3 = getelementptr { double, double }, ptr %tmp1, i64 0, i32 1          ; <ptr> [#uses=1]
        store double %tmp.upgrd.1, ptr %real3
        %tmp.upgrd.2 = getelementptr { double, double }, ptr %tmp, i64 0, i32 0             ; <ptr> [#uses=1]
        %tmp4 = getelementptr { double, double }, ptr %tmp1, i64 0, i32 0           ; <ptr> [#uses=1]
        %tmp5 = load double, ptr %tmp4              ; <double> [#uses=1]
        store double %tmp5, ptr %tmp.upgrd.2
        %tmp6 = getelementptr { double, double }, ptr %tmp, i64 0, i32 1            ; <ptr> [#uses=1]
        %tmp7 = getelementptr { double, double }, ptr %tmp1, i64 0, i32 1           ; <ptr> [#uses=1]
        %tmp8 = load double, ptr %tmp7              ; <double> [#uses=1]
        store double %tmp8, ptr %tmp6
        %tmp.upgrd.4 = getelementptr { i64, i64 }, ptr %tmp, i64 0, i32 0           ; <ptr> [#uses=1]
        %tmp.upgrd.5 = load i64, ptr %tmp.upgrd.4           ; <i64> [#uses=1]
        %tmp10 = getelementptr { i64, i64 }, ptr %tmp, i64 0, i32 1                ; <ptr> [#uses=1]
        %tmp11 = load i64, ptr %tmp10               ; <i64> [#uses=1]
        call void @cexp( ptr sret({ double, double })  %tmp2, i64 %tmp.upgrd.5, i64 %tmp11 )
        %tmp12 = getelementptr { double, double }, ptr %z, i64 0, i32 0             ; <ptr> [#uses=1]
        %tmp13 = getelementptr { double, double }, ptr %tmp2, i64 0, i32 0          ; <ptr> [#uses=1]
        %tmp14 = load double, ptr %tmp13            ; <double> [#uses=1]
        store double %tmp14, ptr %tmp12
        %tmp15 = getelementptr { double, double }, ptr %z, i64 0, i32 1             ; <ptr> [#uses=1]
        %tmp16 = getelementptr { double, double }, ptr %tmp2, i64 0, i32 1          ; <ptr> [#uses=1]
        %tmp17 = load double, ptr %tmp16            ; <double> [#uses=1]
        store double %tmp17, ptr %tmp15
        %tmp18 = getelementptr { double, double }, ptr %z, i64 0, i32 1             ; <ptr> [#uses=1]
        %tmp19 = load double, ptr %tmp18            ; <double> [#uses=1]
        %tmp20 = getelementptr { double, double }, ptr %z, i64 0, i32 0             ; <ptr> [#uses=1]
        %tmp21 = load double, ptr %tmp20            ; <double> [#uses=1]
        %tmp.upgrd.6 = getelementptr [9 x i8], ptr @str, i32 0, i64 0               ; <ptr> [#uses=1]
        %tmp.upgrd.7 = call i32 (ptr, ...) @printf( ptr %tmp.upgrd.6, double %tmp21, double %tmp19 )           ; <i32> [#uses=0]
        br label %finish
finish:
        %retval.upgrd.8 = load i32, ptr %retval             ; <i32> [#uses=1]
        ret i32 %retval.upgrd.8
}

declare void @cexp(ptr sret({ double, double }), i64, i64)

declare i32 @printf(ptr, ...)

