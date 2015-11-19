; RUN: llc < %s -march=arm64 -mcpu=cyclone -enable-misched=false | FileCheck %s
target triple = "arm64-apple-ios7.0.0"

; rdar://13625505
; Here we have 9 fixed integer arguments the 9th argument in on stack, the
; varargs start right after at 8-byte alignment.
define void @fn9(i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %a8, i32 %a9, ...) nounwind noinline ssp {
; CHECK-LABEL: fn9:
; 9th fixed argument
; CHECK: ldr {{w[0-9]+}}, [sp, #64]
; CHECK: add [[ARGS:x[0-9]+]], sp, #72
; CHECK: add {{x[0-9]+}}, [[ARGS]], #8
; First vararg
; CHECK: ldr {{w[0-9]+}}, [sp, #72]
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, #8
; Second vararg
; CHECK: ldr {{w[0-9]+}}, [{{x[0-9]+}}]
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, #8
; Third vararg
; CHECK: ldr {{w[0-9]+}}, [{{x[0-9]+}}]
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %args = alloca i8*, align 8
  %a10 = alloca i32, align 4
  %a11 = alloca i32, align 4
  %a12 = alloca i32, align 4
  store i32 %a1, i32* %1, align 4
  store i32 %a2, i32* %2, align 4
  store i32 %a3, i32* %3, align 4
  store i32 %a4, i32* %4, align 4
  store i32 %a5, i32* %5, align 4
  store i32 %a6, i32* %6, align 4
  store i32 %a7, i32* %7, align 4
  store i32 %a8, i32* %8, align 4
  store i32 %a9, i32* %9, align 4
  %10 = bitcast i8** %args to i8*
  call void @llvm.va_start(i8* %10)
  %11 = va_arg i8** %args, i32
  store i32 %11, i32* %a10, align 4
  %12 = va_arg i8** %args, i32
  store i32 %12, i32* %a11, align 4
  %13 = va_arg i8** %args, i32
  store i32 %13, i32* %a12, align 4
  ret void
}

declare void @llvm.va_start(i8*) nounwind

define i32 @main() nounwind ssp {
; CHECK-LABEL: main:
; CHECK: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp, #16]
; CHECK: str {{x[0-9]+}}, [sp, #8]
; CHECK: str {{w[0-9]+}}, [sp]
  %a1 = alloca i32, align 4
  %a2 = alloca i32, align 4
  %a3 = alloca i32, align 4
  %a4 = alloca i32, align 4
  %a5 = alloca i32, align 4
  %a6 = alloca i32, align 4
  %a7 = alloca i32, align 4
  %a8 = alloca i32, align 4
  %a9 = alloca i32, align 4
  %a10 = alloca i32, align 4
  %a11 = alloca i32, align 4
  %a12 = alloca i32, align 4
  store i32 1, i32* %a1, align 4
  store i32 2, i32* %a2, align 4
  store i32 3, i32* %a3, align 4
  store i32 4, i32* %a4, align 4
  store i32 5, i32* %a5, align 4
  store i32 6, i32* %a6, align 4
  store i32 7, i32* %a7, align 4
  store i32 8, i32* %a8, align 4
  store i32 9, i32* %a9, align 4
  store i32 10, i32* %a10, align 4
  store i32 11, i32* %a11, align 4
  store i32 12, i32* %a12, align 4
  %1 = load i32, i32* %a1, align 4
  %2 = load i32, i32* %a2, align 4
  %3 = load i32, i32* %a3, align 4
  %4 = load i32, i32* %a4, align 4
  %5 = load i32, i32* %a5, align 4
  %6 = load i32, i32* %a6, align 4
  %7 = load i32, i32* %a7, align 4
  %8 = load i32, i32* %a8, align 4
  %9 = load i32, i32* %a9, align 4
  %10 = load i32, i32* %a10, align 4
  %11 = load i32, i32* %a11, align 4
  %12 = load i32, i32* %a12, align 4
  call void (i32, i32, i32, i32, i32, i32, i32, i32, i32, ...) @fn9(i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10, i32 %11, i32 %12)
  ret i32 0
}

;rdar://13668483
@.str = private unnamed_addr constant [4 x i8] c"fmt\00", align 1
define void @foo(i8* %fmt, ...) nounwind {
entry:
; CHECK-LABEL: foo:
; CHECK: orr {{x[0-9]+}}, {{x[0-9]+}}, #0x8
; CHECK: ldr {{w[0-9]+}}, [sp, #48]
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, #15
; CHECK: and x[[ADDR:[0-9]+]], {{x[0-9]+}}, #0xfffffffffffffff0
; CHECK: ldr {{q[0-9]+}}, [x[[ADDR]]]
  %fmt.addr = alloca i8*, align 8
  %args = alloca i8*, align 8
  %vc = alloca i32, align 4
  %vv = alloca <4 x i32>, align 16
  store i8* %fmt, i8** %fmt.addr, align 8
  %args1 = bitcast i8** %args to i8*
  call void @llvm.va_start(i8* %args1)
  %0 = va_arg i8** %args, i32
  store i32 %0, i32* %vc, align 4
  %1 = va_arg i8** %args, <4 x i32>
  store <4 x i32> %1, <4 x i32>* %vv, align 16
  ret void
}

define void @bar(i32 %x, <4 x i32> %y) nounwind {
entry:
; CHECK-LABEL: bar:
; CHECK: str {{q[0-9]+}}, [sp, #16]
; CHECK: str {{x[0-9]+}}, [sp]
  %x.addr = alloca i32, align 4
  %y.addr = alloca <4 x i32>, align 16
  store i32 %x, i32* %x.addr, align 4
  store <4 x i32> %y, <4 x i32>* %y.addr, align 16
  %0 = load i32, i32* %x.addr, align 4
  %1 = load <4 x i32>, <4 x i32>* %y.addr, align 16
  call void (i8*, ...) @foo(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i32 %0, <4 x i32> %1)
  ret void
}

; rdar://13668927
; When passing 16-byte aligned small structs as vararg, make sure the caller
; side is 16-byte aligned on stack.
%struct.s41 = type { i32, i16, i32, i16 }
define void @foo2(i8* %fmt, ...) nounwind {
entry:
; CHECK-LABEL: foo2:
; CHECK: orr {{x[0-9]+}}, {{x[0-9]+}}, #0x8
; CHECK: ldr {{w[0-9]+}}, [sp, #48]
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, #15
; CHECK: and x[[ADDR:[0-9]+]], {{x[0-9]+}}, #0xfffffffffffffff0
; CHECK: ldr {{q[0-9]+}}, [x[[ADDR]]]
  %fmt.addr = alloca i8*, align 8
  %args = alloca i8*, align 8
  %vc = alloca i32, align 4
  %vs = alloca %struct.s41, align 16
  store i8* %fmt, i8** %fmt.addr, align 8
  %args1 = bitcast i8** %args to i8*
  call void @llvm.va_start(i8* %args1)
  %0 = va_arg i8** %args, i32
  store i32 %0, i32* %vc, align 4
  %ap.cur = load i8*, i8** %args
  %1 = getelementptr i8, i8* %ap.cur, i32 15
  %2 = ptrtoint i8* %1 to i64
  %3 = and i64 %2, -16
  %ap.align = inttoptr i64 %3 to i8*
  %ap.next = getelementptr i8, i8* %ap.align, i32 16
  store i8* %ap.next, i8** %args
  %4 = bitcast i8* %ap.align to %struct.s41*
  %5 = bitcast %struct.s41* %vs to i8*
  %6 = bitcast %struct.s41* %4 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %5, i8* %6, i64 16, i1 false)
  ret void
}
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind

define void @bar2(i32 %x, i128 %s41.coerce) nounwind {
entry:
; CHECK-LABEL: bar2:
; CHECK: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp, #16]
; CHECK: str {{x[0-9]+}}, [sp]
  %x.addr = alloca i32, align 4
  %s41 = alloca %struct.s41, align 16
  store i32 %x, i32* %x.addr, align 4
  %0 = bitcast %struct.s41* %s41 to i128*
  store i128 %s41.coerce, i128* %0, align 1
  %1 = load i32, i32* %x.addr, align 4
  %2 = bitcast %struct.s41* %s41 to i128*
  %3 = load i128, i128* %2, align 1
  call void (i8*, ...) @foo2(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i32 %1, i128 %3)
  ret void
}
