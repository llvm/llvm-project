; RUN: llc -O2 -mtriple powerpc64-ibm-aix-xcoff -stop-after=machine-cp -verify-machineinstrs < %s | \
; RUN: FileCheck --check-prefix=64BIT %s

; RUN: llc -O2 -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec \
; RUN: -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefix=ASM64 %s

  define i32 @int_va_arg(i32 %a, ...) local_unnamed_addr  {
  entry:
    %arg1 = alloca i8*, align 8
    %arg2 = alloca i8*, align 8
    %0 = bitcast i8** %arg1 to i8*
    call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0)
    %1 = bitcast i8** %arg2 to i8*
    call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1)
    call void @llvm.va_start(i8* nonnull %0)
    call void @llvm.va_copy(i8* nonnull %1, i8* nonnull %0)
    %2 = va_arg i8** %arg1, i32
    %add = add nsw i32 %2, %a
    %3 = va_arg i8** %arg2, i32
    %mul = shl i32 %3, 1
    %add3 = add nsw i32 %add, %mul
    call void @llvm.va_end(i8* nonnull %0)
    call void @llvm.va_end(i8* nonnull %1)
    call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1)
    call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0)
    ret i32 %add3
  }

  declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)
  declare void @llvm.va_start(i8*)
  declare void @llvm.va_copy(i8*, i8*)
  declare void @llvm.va_end(i8*)
  declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

; 64BIT-LABEL:   name:            int_va_arg
; 64BIT-LABEL:   liveins:
; 64BIT-DAG:     - { reg: '$x3', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$x4', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$x5', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$x6', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$x7', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$x8', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$x9', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$x10', virtual-reg: '' }

; 64BIT-LABEL:   fixedStack:
; 64BIT-DAG:     - { id: 0, type: default, offset: 56, size: 8

; 64BIT-LABEL:   stack:
; 64BIT-DAG:     - { id: 0, name: arg1, type: default, offset: 0, size: 8
; 64BIT-DAG:     - { id: 1, name: arg2, type: default, offset: 0, size: 8

; 64BIT-LABEL:   body:             |
; 64BIT-DAG:     bb.0.entry:
; 64BIT-DAG:     liveins: $x3, $x4, $x5, $x6, $x7, $x8, $x9, $x10
; 64BIT-DAG:     STD killed renamable $x4, 0, %fixed-stack.0 :: (store (s64) into %fixed-stack.0)
; 64BIT-DAG:     STD killed renamable $x5, 8, %fixed-stack.0 :: (store (s64) into %fixed-stack.0 + 8)
; 64BIT-DAG:     STD killed renamable $x6, 16, %fixed-stack.0 :: (store (s64))
; 64BIT-DAG:     STD killed renamable $x7, 24, %fixed-stack.0 :: (store (s64))
; 64BIT-DAG:     STD killed renamable $x8, 32, %fixed-stack.0 :: (store (s64))
; 64BIT-DAG:     STD killed renamable $x9, 40, %fixed-stack.0 :: (store (s64))
; 64BIT-DAG:     STD killed renamable $x10, 48, %fixed-stack.0 :: (store (s64))
; 64BIT-DAG:     renamable $x11 = ADDI8 %fixed-stack.0, 0
; 64BIT-DAG:     STD renamable $x11, 0, %stack.1.arg2 :: (store (s64) into %ir.1)
; 64BIT-DAG:     renamable $x6 = LD 0, %stack.1.arg2 :: (load (s64) from %ir.arg2)
; 64BIT-DAG:     renamable $x9 = ADDI8 renamable $x6, 4
; 64BIT-DAG:     renamable $x7 = ADDI8 %fixed-stack.0, 4
; 64BIT-DAG:     renamable $r8 = LWZ 0, %fixed-stack.0 :: (load (s32) from %fixed-stack.0, align 8)
; 64BIT-DAG:     STD killed renamable $x11, 0, %stack.0.arg1 :: (store (s64) into %ir.0)
; 64BIT-DAG:     STD killed renamable $x7, 0, %stack.0.arg1 :: (store (s64) into %ir.arg1)
; 64BIT-DAG:     STD killed renamable $x9, 0, %stack.1.arg2 :: (store (s64) into %ir.arg2)
; 64BIT-DAG:     renamable $r4 = LWZ 0, killed renamable $x6 :: (load (s32))
; 64BIT-DAG:     renamable $r3 = nsw ADD4 killed renamable $r8, renamable $r3, implicit killed $x3
; 64BIT-DAG:     renamable $r4 = RLWINM killed renamable $r4, 1, 0, 30
; 64BIT-DAG:     renamable $r3 = nsw ADD4 killed renamable $r3, killed renamable $r4, implicit-def $x3
; 64BIT-DAG:     BLR8 implicit $lr8, implicit $rm, implicit $x3

; ASM64-LABEL:   .int_va_arg:
; ASM64-DAG:     std 4, 56(1)
; ASM64-DAG:     addi 4, 1, 56
; ASM64-DAG:     std 4, -16(1)
; ASM64-DAG:     std 4, -8(1)
; ASM64-DAG:     ld 4, -16(1)
; ASM64-DAG:     std 5, 64(1)
; ASM64-DAG:     addi 5, 1, 60
; ASM64-DAG:     std 5, -8(1)
; ASM64-DAG:     addi 5, 4, 4
; ASM64-DAG:     std 6, 72(1)
; ASM64-DAG:     std 7, 80(1)
; ASM64-DAG:     std 8, 88(1)
; ASM64-DAG:     std 9, 96(1)
; ASM64-DAG:     std 10, 104(1)
; ASM64-DAG:     std 5, -16(1)
; ASM64-DAG:     lwz 11, 56(1)
; ASM64-DAG:     lwz 4, 0(4)
; ASM64-DAG:     add 3, 11, 3
; ASM64-DAG:     slwi 4, 4, 1
; ASM64-DAG:     add 3, 3, 4
; ASM64-DAG:     blr

  define i32 @int_stack_va_arg(i32 %one, i32 %two, i32 %three, i32 %four, i32 %five, i32 %six, i32 %seven, i32 %eight, ...) local_unnamed_addr {
  entry:
    %arg1 = alloca i8*, align 8
    %arg2 = alloca i8*, align 8
    %0 = bitcast i8** %arg1 to i8*
    call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0)
    %1 = bitcast i8** %arg2 to i8*
    call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1)
    call void @llvm.va_start(i8* nonnull %0)
    call void @llvm.va_copy(i8* nonnull %1, i8* nonnull %0)
    %add = add nsw i32 %two, %one
    %add2 = add nsw i32 %add, %three
    %add3 = add nsw i32 %add2, %four
    %add4 = add nsw i32 %add3, %five
    %add5 = add nsw i32 %add4, %six
    %add6 = add nsw i32 %add5, %seven
    %add7 = add nsw i32 %add6, %eight
    %2 = va_arg i8** %arg1, i32
    %add8 = add nsw i32 %add7, %2
    %3 = va_arg i8** %arg2, i32
    %mul = shl i32 %3, 1
    %add10 = add nsw i32 %add8, %mul
    call void @llvm.va_end(i8* nonnull %0)
    call void @llvm.va_end(i8* nonnull %1)
    call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1)
    call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0)
    ret i32 %add10
  }

; 64BIT-LABEL:    name:            int_stack_va_arg
; 64BIT-LABEL:    liveins:
; 64BIT-DAG:       - { reg: '$x3', virtual-reg: '' }
; 64BIT-DAG:       - { reg: '$x4', virtual-reg: '' }
; 64BIT-DAG:       - { reg: '$x5', virtual-reg: '' }
; 64BIT-DAG:       - { reg: '$x6', virtual-reg: '' }
; 64BIT-DAG:       - { reg: '$x7', virtual-reg: '' }
; 64BIT-DAG:       - { reg: '$x8', virtual-reg: '' }
; 64BIT-DAG:       - { reg: '$x9', virtual-reg: '' }
; 64BIT-DAG:       - { reg: '$x10', virtual-reg: '' }

; 64BIT-LABEL:   fixedStack:
; 64BIT-DAG:     - { id: 0, type: default, offset: 112, size: 8, alignment: 16, stack-id: default,

; 64BIT-LABEL:   stack:
; 64BIT-DAG:     - { id: 0, name: arg1, type: default, offset: 0, size: 8, alignment: 8,
; 64BIT-DAG:     - { id: 1, name: arg2, type: default, offset: 0, size: 8, alignment: 8,

; 64BIT-LABEL:   body:             |
; 64BIT-DAG:     liveins: $x3, $x4, $x5, $x6, $x7, $x8, $x9, $x10
; 64BIT-DAG:     renamable $r11 = LWZ 0, %fixed-stack.0 :: (load (s32) from %fixed-stack.0, align 16)
; 64BIT-DAG:     renamable $r3 = nsw ADD4 renamable $r4, renamable $r3, implicit killed $x3, implicit killed $x4
; 64BIT-DAG:     renamable $r3 = nsw ADD4 killed renamable $r3, renamable $r5, implicit killed $x5
; 64BIT-DAG:     renamable $r3 = nsw ADD4 killed renamable $r3, renamable $r6, implicit killed $x6
; 64BIT-DAG:     renamable $r3 = nsw ADD4 killed renamable $r3, renamable $r7, implicit killed $x7
; 64BIT-DAG:     renamable $r3 = nsw ADD4 killed renamable $r3, renamable $r8, implicit killed $x8
; 64BIT-DAG:     renamable $r3 = nsw ADD4 killed renamable $r3, renamable $r9, implicit killed $x9
; 64BIT-DAG:     renamable $r3 = nsw ADD4 killed renamable $r3, renamable $r10, implicit killed $x10
; 64BIT-DAG:     renamable $r3 = nsw ADD4 killed renamable $r3, renamable $r11
; 64BIT-DAG:     renamable $r3 = nsw ADD4 killed renamable $r3, killed renamable $r4, implicit-def $x3
; 64BIT-DAG:     BLR8 implicit $lr8, implicit $rm, implicit $x3

; ASM64-LABEL:   .int_stack_va_arg:
; ASM64-DAG:     add 3, 4, 3
; ASM64-DAG:     add 3, 3, 5
; ASM64-DAG:     add 3, 3, 6
; ASM64-DAG:     add 3, 3, 7
; ASM64-DAG:     add 3, 3, 8
; ASM64-DAG:     add 3, 3, 9
; ASM64-DAG:     add 3, 3, 10
; ASM64-DAG:     lwz 11, 112(1)
; ASM64-DAG:     slwi 4, 11, 1
; ASM64-DAG:     add 3, 3, 11
; ASM64-DAG:     add 3, 3, 4
; ASM64-DAG:     blr

  define double @double_va_arg(double %a, ...) local_unnamed_addr  {
  entry:
    %arg1 = alloca i8*, align 8
    %arg2 = alloca i8*, align 8
    %0 = bitcast i8** %arg1 to i8*
    call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0)
    %1 = bitcast i8** %arg2 to i8*
    call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1)
    call void @llvm.va_start(i8* nonnull %0)
    call void @llvm.va_copy(i8* nonnull %1, i8* nonnull %0)
    %2 = va_arg i8** %arg1, double
    %add = fadd double %2, %a
    %3 = va_arg i8** %arg2, double
    %mul = fmul double %3, 2.000000e+00
    %add3 = fadd double %add, %mul
    call void @llvm.va_end(i8* nonnull %0)
    call void @llvm.va_end(i8* nonnull %1)
    call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1)
    call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0)
    ret double %add3
  }

; 64BIT-LABEL:   name:            double_va_arg
; 64BIT-LABEL:   liveins:
; 64BIT-DAG:     - { reg: '$f1', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$x4', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$x5', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$x6', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$x7', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$x8', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$x9', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$x10', virtual-reg: '' }

; 64BIT-LABEL:   fixedStack:
; 64BIT-DAG:     - { id: 0, type: default, offset: 56, size: 8

; 64BIT-LABEL:   stack:
; 64BIT-DAG:     - { id: 0, name: arg1, type: default, offset: 0, size: 8
; 64BIT-DAG:     - { id: 1, name: arg2, type: default, offset: 0, size: 8

; 64BIT-LABEL:   body:             |
; 64BIT-DAG:     liveins: $f1, $x4, $x5, $x6, $x7, $x8, $x9, $x10
; 64BIT-DAG:     renamable $x3 = ADDI8 %fixed-stack.0, 0
; 64BIT-DAG:     STD killed renamable $x4, 0, %fixed-stack.0 :: (store (s64) into %fixed-stack.0)
; 64BIT-DAG:     STD killed renamable $x5, 8, %fixed-stack.0 :: (store (s64) into %fixed-stack.0 + 8)
; 64BIT-DAG:     STD killed renamable $x6, 16, %fixed-stack.0 :: (store (s64))
; 64BIT-DAG:     STD killed renamable $x7, 24, %fixed-stack.0 :: (store (s64))
; 64BIT-DAG:     STD killed renamable $x8, 32, %fixed-stack.0 :: (store (s64))
; 64BIT-DAG:     STD killed renamable $x9, 40, %fixed-stack.0 :: (store (s64))
; 64BIT-DAG:     STD killed renamable $x10, 48, %fixed-stack.0 :: (store (s64))
; 64BIT-DAG:     STD renamable $x3, 0, %stack.1.arg2 :: (store (s64) into %ir.1)
; 64BIT-DAG:     renamable $x6 = LD 0, %stack.1.arg2 :: (load (s64) from %ir.arg2)
; 64BIT-DAG:     renamable $x7 = ADDI8 %fixed-stack.0, 8
; 64BIT-DAG:     STD killed renamable $x3, 0, %stack.0.arg1 :: (store (s64) into %ir.0)
; 64BIT-DAG:     STD killed renamable $x7, 0, %stack.0.arg1 :: (store (s64) into %ir.arg1)
; 64BIT-DAG:     renamable $f0 = LFD 0, %fixed-stack.0 :: (load (s64))
; 64BIT-DAG:     renamable $x3 = ADDI8 renamable $x6, 8
; 64BIT-DAG:     STD killed renamable $x3, 0, %stack.1.arg2 :: (store (s64) into %ir.arg2)
; 64BIT-DAG:     renamable $f2 = LFD 0, killed renamable $x6 :: (load (s64))
; 64BIT-DAG:     renamable $f0 = nofpexcept FADD killed renamable $f0, killed renamable $f1, implicit $rm
; 64BIT-DAG:     renamable $f1 = nofpexcept FADD killed renamable $f2, renamable $f2, implicit $rm
; 64BIT-DAG:     renamable $f1 = nofpexcept FADD killed renamable $f0, killed renamable $f1, implicit $rm
; 64BIT-DAG:     BLR8 implicit $lr8, implicit $rm, implicit $f1

; ASM64-LABEL:  .double_va_arg:
; ASM64-DAG:    addi 3, 1, 56
; ASM64-DAG:    std 4, 56(1)
; ASM64-DAG:    std 3, -8(1)
; ASM64-DAG:    std 3, -16(1)
; ASM64-DAG:    addi 3, 1, 64
; ASM64-DAG:    std 3, -8(1)
; ASM64-DAG:    ld 3, -16(1)
; ASM64-DAG:    lfd 0, 56(1)
; ASM64-DAG:    addi 4, 3, 8
; ASM64-DAG:    std 5, 64(1)
; ASM64-DAG:    fadd 0, 0, 1
; ASM64-DAG:    std 6, 72(1)
; ASM64-DAG:    std 7, 80(1)
; ASM64-DAG:    std 8, 88(1)
; ASM64-DAG:    std 9, 96(1)
; ASM64-DAG:    std 10, 104(1)
; ASM64-DAG:    std 4, -16(1)
; ASM64-DAG:    lfd 1, 0(3)
; ASM64-DAG:    fadd 1, 1, 1
; ASM64-DAG:    fadd 1, 0, 1
; ASM64-DAG:    blr

  define double @double_stack_va_arg(double %one, double %two, double %three, double %four, double %five, double %six, double %seven, double %eight, double %nine, double %ten, double %eleven, double %twelve, double %thirteen, ...) local_unnamed_addr  {
  entry:
    %arg1 = alloca i8*, align 8
    %arg2 = alloca i8*, align 8
    %0 = bitcast i8** %arg1 to i8*
    call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0)
    %1 = bitcast i8** %arg2 to i8*
    call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1)
    call void @llvm.va_start(i8* nonnull %0)
    call void @llvm.va_copy(i8* nonnull %1, i8* nonnull %0)
    %add = fadd double %one, %two
    %add2 = fadd double %add, %three
    %add3 = fadd double %add2, %four
    %add4 = fadd double %add3, %five
    %add5 = fadd double %add4, %six
    %add6 = fadd double %add5, %seven
    %add7 = fadd double %add6, %eight
    %add8 = fadd double %add7, %nine
    %add9 = fadd double %add8, %ten
    %add10 = fadd double %add9, %eleven
    %add11 = fadd double %add10, %twelve
    %add12 = fadd double %add11, %thirteen
    %2 = va_arg i8** %arg1, double
    %add13 = fadd double %add12, %2
    %3 = va_arg i8** %arg2, double
    %mul = fmul double %3, 2.000000e+00
    %add15 = fadd double %add13, %mul
    call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1)
    call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0)
    ret double %add15
  }


; 64BIT-LABEL:   name:            double_stack_va_arg
; 64BIT-LABEL:   liveins:
; 64BIT-DAG:     - { reg: '$f1', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$f2', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$f3', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$f4', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$f5', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$f6', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$f7', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$f8', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$f9', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$f10', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$f11', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$f12', virtual-reg: '' }
; 64BIT-DAG:     - { reg: '$f13', virtual-reg: '' }

; 64BIT-LABEL:   fixedStack:
; 64BIT-DAG:       - { id: 0, type: default, offset: 152, size: 8

; 64BIT-LABEL:   stack:
; 64BIT-DAG:       - { id: 0, name: arg1, type: default, offset: 0, size: 8
; 64BIT-DAG:       - { id: 1, name: arg2, type: default, offset: 0, size: 8

; 64BIT-LABEL:     body:             |
; 64BIT-DAG:       liveins: $f1, $f2, $f3, $f4, $f5, $f6, $f7, $f8, $f9, $f10, $f11, $f12, $f13
; 64BIT-DAG:       renamable $f0 = LFD 0, %fixed-stack.0 :: (load (s64))
; 64BIT-DAG:       renamable $f1 = nofpexcept FADD killed renamable $f1, killed renamable $f2, implicit $rm
; 64BIT-DAG:       renamable $f1 = nofpexcept FADD killed renamable $f1, killed renamable $f3, implicit $rm
; 64BIT-DAG:       renamable $f1 = nofpexcept FADD killed renamable $f1, killed renamable $f4, implicit $rm
; 64BIT-DAG:       renamable $f1 = nofpexcept FADD killed renamable $f1, killed renamable $f5, implicit $rm
; 64BIT-DAG:       renamable $f1 = nofpexcept FADD killed renamable $f1, killed renamable $f6, implicit $rm
; 64BIT-DAG:       renamable $f1 = nofpexcept FADD killed renamable $f1, killed renamable $f7, implicit $rm
; 64BIT-DAG:       renamable $f1 = nofpexcept FADD killed renamable $f1, killed renamable $f8, implicit $rm
; 64BIT-DAG:       renamable $f1 = nofpexcept FADD killed renamable $f1, killed renamable $f9, implicit $rm
; 64BIT-DAG:       renamable $f1 = nofpexcept FADD killed renamable $f1, killed renamable $f10, implicit $rm
; 64BIT-DAG:       renamable $f1 = nofpexcept FADD killed renamable $f1, killed renamable $f11, implicit $rm
; 64BIT-DAG:       renamable $f1 = nofpexcept FADD killed renamable $f1, killed renamable $f12, implicit $rm
; 64BIT-DAG:       renamable $f1 = nofpexcept FADD killed renamable $f1, killed renamable $f13, implicit $rm
; 64BIT-DAG:       renamable $f1 = nofpexcept FADD killed renamable $f1, renamable $f0, implicit $rm
; 64BIT-DAG:       renamable $f0 = nofpexcept FADD killed renamable $f0, renamable $f0, implicit $rm
; 64BIT-DAG:       renamable $f1 = nofpexcept FADD killed renamable $f1, killed renamable $f0, implicit $rm
; 64BIT-DAG:       BLR8 implicit $lr8, implicit $rm, implicit $f1

; ASM64-LABEL:   .double_stack_va_arg:
; ASM64-DAG:     fadd 1, 1, 2
; ASM64-DAG:     fadd 1, 1, 3
; ASM64-DAG:     fadd 1, 1, 4
; ASM64-DAG:     fadd 1, 1, 5
; ASM64-DAG:     fadd 1, 1, 6
; ASM64-DAG:     fadd 1, 1, 7
; ASM64-DAG:     fadd 1, 1, 8
; ASM64-DAG:     fadd 1, 1, 9
; ASM64-DAG:     fadd 1, 1, 10
; ASM64-DAG:     fadd 1, 1, 11
; ASM64-DAG:     fadd 1, 1, 12
; ASM64-DAG:     fadd 1, 1, 13
; ASM64-DAG:     lfd 0, 152(1)
; ASM64-DAG:     fadd 1, 1, 0
; ASM64-DAG:     fadd 0, 0, 0
; ASM64-DAG:     fadd 1, 1, 0
; ASM64-DAG:     blr
