; RUN: llc -march=mipsel -pre-RA-sched=source < %s | FileCheck %s

; All test functions do the same thing - they return the first variable
; argument.

; All CHECK's do the same thing - they check whether variable arguments from
; registers are placed on correct stack locations, and whether the first
; variable argument is returned from the correct stack location.


declare void @llvm.va_start(i8*) nounwind
declare void @llvm.va_end(i8*) nounwind

; return int
define i32 @va1(i32 %a, ...) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %ap = alloca i8*, align 4
  %b = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %0 = va_arg i8** %ap, i32
  store i32 %0, i32* %b, align 4
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_end(i8* %ap2)
  %tmp = load i32, i32* %b, align 4
  ret i32 %tmp

; CHECK-LABEL: va1:
; CHECK: addiu   $sp, $sp, -16
; CHECK: sw      $5, 20($sp)
; CHECK: sw      $7, 28($sp)
; CHECK: sw      $6, 24($sp)
; CHECK: lw      $2, 20($sp)
}

; check whether the variable double argument will be accessed from the 8-byte
; aligned location (i.e. whether the address is computed by adding 7 and
; clearing lower 3 bits)
define double @va2(i32 %a, ...) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %ap = alloca i8*, align 4
  %b = alloca double, align 8
  store i32 %a, i32* %a.addr, align 4
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %0 = va_arg i8** %ap, double
  store double %0, double* %b, align 8
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_end(i8* %ap2)
  %tmp = load double, double* %b, align 8
  ret double %tmp

; CHECK-LABEL: va2:
; CHECK: addiu   $sp, $sp, -16
; CHECK: sw      $7, 28($sp)
; CHECK: sw      $6, 24($sp)
; CHECK: sw      $5, 20($sp)
; CHECK: addiu   $[[R0:[0-9]+]], $sp, 20
; CHECK: addiu   $[[R1:[0-9]+]], $[[R0]], 7
; CHECK: addiu   $[[R2:[0-9]+]], $zero, -8
; CHECK: and     $[[R3:[0-9]+]], $[[R1]], $[[R2]]
; CHECK: ldc1    $f0, 0($[[R3]])
}

; int
define i32 @va3(double %a, ...) nounwind {
entry:
  %a.addr = alloca double, align 8
  %ap = alloca i8*, align 4
  %b = alloca i32, align 4
  store double %a, double* %a.addr, align 8
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %0 = va_arg i8** %ap, i32
  store i32 %0, i32* %b, align 4
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_end(i8* %ap2)
  %tmp = load i32, i32* %b, align 4
  ret i32 %tmp

; CHECK-LABEL: va3:
; CHECK: addiu   $sp, $sp, -16
; CHECK: sw      $6, 24($sp)
; CHECK: sw      $7, 28($sp)
; CHECK: lw      $2, 24($sp)
}

; double
define double @va4(double %a, ...) nounwind {
entry:
  %a.addr = alloca double, align 8
  %ap = alloca i8*, align 4
  %b = alloca double, align 8
  store double %a, double* %a.addr, align 8
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %0 = va_arg i8** %ap, double
  store double %0, double* %b, align 8
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_end(i8* %ap2)
  %tmp = load double, double* %b, align 8
  ret double %tmp

; CHECK-LABEL: va4:
; CHECK: addiu   $sp, $sp, -24
; CHECK: sw      $7, 36($sp)
; CHECK: sw      $6, 32($sp)
; CHECK: addiu   ${{[0-9]+}}, $sp, 32
; CHECK: ldc1    $f0, 32($sp)
}

; int
define i32 @va5(i32 %a, i32 %b, i32 %c, ...) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %c.addr = alloca i32, align 4
  %ap = alloca i8*, align 4
  %d = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 %b, i32* %b.addr, align 4
  store i32 %c, i32* %c.addr, align 4
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %0 = va_arg i8** %ap, i32
  store i32 %0, i32* %d, align 4
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_end(i8* %ap2)
  %tmp = load i32, i32* %d, align 4
  ret i32 %tmp

; CHECK-LABEL: va5:
; CHECK: addiu   $sp, $sp, -24
; CHECK: sw      $7, 36($sp)
; CHECK: lw      $2, 36($sp)
}

; double
define double @va6(i32 %a, i32 %b, i32 %c, ...) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %c.addr = alloca i32, align 4
  %ap = alloca i8*, align 4
  %d = alloca double, align 8
  store i32 %a, i32* %a.addr, align 4
  store i32 %b, i32* %b.addr, align 4
  store i32 %c, i32* %c.addr, align 4
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %0 = va_arg i8** %ap, double
  store double %0, double* %d, align 8
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_end(i8* %ap2)
  %tmp = load double, double* %d, align 8
  ret double %tmp

; CHECK-LABEL: va6:
; CHECK: addiu   $sp, $sp, -24
; CHECK: sw      $7, 36($sp)
; CHECK: addiu   $[[R0:[0-9]+]], $sp, 36
; CHECK: addiu   $[[R1:[0-9]+]], $[[R0]], 7
; CHECK: addiu   $[[R2:[0-9]+]], $zero, -8
; CHECK: and     $[[R3:[0-9]+]], $[[R1]], $[[R2]]
; CHECK: ldc1    $f0, 0($[[R3]])
}

; int
define i32 @va7(i32 %a, double %b, ...) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca double, align 8
  %ap = alloca i8*, align 4
  %c = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store double %b, double* %b.addr, align 8
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %0 = va_arg i8** %ap, i32
  store i32 %0, i32* %c, align 4
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_end(i8* %ap2)
  %tmp = load i32, i32* %c, align 4
  ret i32 %tmp

; CHECK-LABEL: va7:
; CHECK: addiu   $sp, $sp, -24
; CHECK: lw      $2, 40($sp)
}

; double
define double @va8(i32 %a, double %b, ...) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca double, align 8
  %ap = alloca i8*, align 4
  %c = alloca double, align 8
  store i32 %a, i32* %a.addr, align 4
  store double %b, double* %b.addr, align 8
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %0 = va_arg i8** %ap, double
  store double %0, double* %c, align 8
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_end(i8* %ap2)
  %tmp = load double, double* %c, align 8
  ret double %tmp

; CHECK-LABEL: va8:
; CHECK: addiu   $sp, $sp, -32
; CHECK: addiu   ${{[0-9]+}}, $sp, 48
; CHECK: ldc1    $f0, 48($sp)
}

; int
define i32 @va9(double %a, double %b, i32 %c, ...) nounwind {
entry:
  %a.addr = alloca double, align 8
  %b.addr = alloca double, align 8
  %c.addr = alloca i32, align 4
  %ap = alloca i8*, align 4
  %d = alloca i32, align 4
  store double %a, double* %a.addr, align 8
  store double %b, double* %b.addr, align 8
  store i32 %c, i32* %c.addr, align 4
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %0 = va_arg i8** %ap, i32
  store i32 %0, i32* %d, align 4
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_end(i8* %ap2)
  %tmp = load i32, i32* %d, align 4
  ret i32 %tmp

; CHECK-LABEL: va9:
; CHECK: addiu   $sp, $sp, -24
; CHECK: lw      $2, 44($sp)
}

; double
define double @va10(double %a, double %b, i32 %c, ...) nounwind {
entry:
  %a.addr = alloca double, align 8
  %b.addr = alloca double, align 8
  %c.addr = alloca i32, align 4
  %ap = alloca i8*, align 4
  %d = alloca double, align 8
  store double %a, double* %a.addr, align 8
  store double %b, double* %b.addr, align 8
  store i32 %c, i32* %c.addr, align 4
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %0 = va_arg i8** %ap, double
  store double %0, double* %d, align 8
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_end(i8* %ap2)
  %tmp = load double, double* %d, align 8
  ret double %tmp

; CHECK-LABEL: va10:
; CHECK: addiu   $sp, $sp, -32
; CHECK: addiu   $[[R0:[0-9]+]], $sp, 52
; CHECK: addiu   $[[R1:[0-9]+]], $[[R0]], 7
; CHECK: addiu   $[[R2:[0-9]+]], $zero, -8
; CHECK: and     $[[R3:[0-9]+]], $[[R1]], $[[R2]]
; CHECK: ldc1    $f0, 0($[[R3]])
}
