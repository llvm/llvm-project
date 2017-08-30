; RUN: llc -mtriple=i386-linux-gnu   -mattr=+sse2 -global-isel -stop-after=irtranslator < %s -o - | FileCheck %s --check-prefix=ALL --check-prefix=X32
; RUN: llc -mtriple=x86_64-linux-gnu              -global-isel -stop-after=irtranslator < %s -o - | FileCheck %s --check-prefix=ALL --check-prefix=X64

@a1_8bit = external global i8
@a7_8bit = external global i8
@a8_8bit = external global i8

define i8 @test_i8_args_8(i8 %arg1, i8 %arg2, i8 %arg3, i8 %arg4,
		                      i8 %arg5, i8 %arg6, i8 %arg7, i8 %arg8) {

; ALL-LABEL: name:            test_i8_args_8

; X64: fixedStack:
; X64:  id: [[STACK8:[0-9]+]], type: default, offset: 8, size: 1, alignment: 8,
; X64-NEXT: isImmutable: true,

; X64:  id: [[STACK0:[0-9]+]], type: default, offset: 0, size: 1, alignment: 16,
; X64-NEXT: isImmutable: true,

; X64: liveins: %ecx, %edi, %edx, %esi, %r8d, %r9d
; X64:      [[ARG1:%[0-9]+]](s8) = COPY %edi
; X64-NEXT: %{{[0-9]+}}(s8) = COPY %esi
; X64-NEXT: %{{[0-9]+}}(s8) = COPY %edx
; X64-NEXT: %{{[0-9]+}}(s8) = COPY %ecx
; X64-NEXT: %{{[0-9]+}}(s8) = COPY %r8d
; X64-NEXT: %{{[0-9]+}}(s8) = COPY %r9d
; X64-NEXT: [[ARG7_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK0]]
; X64-NEXT: [[ARG7:%[0-9]+]](s8) = G_LOAD [[ARG7_ADDR]](p0) :: (invariant load 1 from %fixed-stack.[[STACK0]], align 0)
; X64-NEXT: [[ARG8_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK8]]
; X64-NEXT: [[ARG8:%[0-9]+]](s8) = G_LOAD [[ARG8_ADDR]](p0) :: (invariant load 1 from %fixed-stack.[[STACK8]], align 0)

; X32: fixedStack:
; X32:  id: [[STACK28:[0-9]+]], type: default, offset: 28, size: 1, alignment: 4,
; X32-NEXT: isImmutable: true,

; X32:  id: [[STACK24:[0-9]+]], type: default, offset: 24, size: 1, alignment: 8,
; X32-NEXT: isImmutable: true,

; X32:  id: [[STACK20:[0-9]+]], type: default, offset: 20, size: 1, alignment: 4,
; X32-NEXT: isImmutable: true,

; X32:  id: [[STACK16:[0-9]+]], type: default, offset: 16, size: 1, alignment: 16,
; X32-NEXT: isImmutable: true,

; X32:  id: [[STACK12:[0-9]+]], type: default, offset: 12, size: 1, alignment: 4,
; X32-NEXT: isImmutable: true,

; X32:  id: [[STACK8:[0-9]+]], type: default, offset: 8, size: 1, alignment: 8,
;X32-NEXT: isImmutable: true,

; X32:  id: [[STACK4:[0-9]+]], type: default, offset: 4, size: 1, alignment: 4,
; X32-NEXT: isImmutable: true,

; X32:  id: [[STACK0:[0-9]+]], type: default, offset: 0, size: 1, alignment: 16,
; X32-NEXT: isImmutable: true,

; X32:       [[ARG1_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK0]]
; X32-NEXT:  [[ARG1:%[0-9]+]](s8) = G_LOAD [[ARG1_ADDR]](p0) :: (invariant load 1 from %fixed-stack.[[STACK0]], align 0)
; X32-NEXT:  [[ARG2_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK4]]
; X32-NEXT:  [[ARG2:%[0-9]+]](s8) = G_LOAD [[ARG2_ADDR]](p0) :: (invariant load 1 from %fixed-stack.[[STACK4]], align 0)
; X32-NEXT:  [[ARG3_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK8]]
; X32-NEXT:  [[ARG3:%[0-9]+]](s8) = G_LOAD [[ARG3_ADDR]](p0) :: (invariant load 1 from %fixed-stack.[[STACK8]], align 0)
; X32-NEXT:  [[ARG4_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK12]]
; X32-NEXT:  [[ARG4:%[0-9]+]](s8) = G_LOAD [[ARG4_ADDR]](p0) :: (invariant load 1 from %fixed-stack.[[STACK12]], align 0)
; X32-NEXT:  [[ARG5_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK16]]
; X32-NEXT:  [[ARG5:%[0-9]+]](s8) = G_LOAD [[ARG5_ADDR]](p0) :: (invariant load 1 from %fixed-stack.[[STACK16]], align 0)
; X32-NEXT:  [[ARG6_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK20]]
; X32-NEXT:  [[ARG6:%[0-9]+]](s8) = G_LOAD [[ARG6_ADDR]](p0) :: (invariant load 1 from %fixed-stack.[[STACK20]], align 0)
; X32-NEXT:  [[ARG7_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK24]]
; X32-NEXT:  [[ARG7:%[0-9]+]](s8) = G_LOAD [[ARG7_ADDR]](p0) :: (invariant load 1 from %fixed-stack.[[STACK24]], align 0)
; X32-NEXT:  [[ARG8_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK28]]
; X32-NEXT:  [[ARG8:%[0-9]+]](s8) = G_LOAD [[ARG8_ADDR]](p0) :: (invariant load 1 from %fixed-stack.[[STACK28]], align 0)

; ALL-NEXT:  [[GADDR_A1:%[0-9]+]](p0) = G_GLOBAL_VALUE @a1_8bit
; ALL-NEXT:  [[GADDR_A7:%[0-9]+]](p0) = G_GLOBAL_VALUE @a7_8bit
; ALL-NEXT:  [[GADDR_A8:%[0-9]+]](p0) = G_GLOBAL_VALUE @a8_8bit
; ALL-NEXT:  G_STORE [[ARG1]](s8), [[GADDR_A1]](p0) :: (store 1 into @a1_8bit)
; ALL-NEXT:  G_STORE [[ARG7]](s8), [[GADDR_A7]](p0) :: (store 1 into @a7_8bit)
; ALL-NEXT:  G_STORE [[ARG8]](s8), [[GADDR_A8]](p0) :: (store 1 into @a8_8bit)
; ALL-NEXT:  %al = COPY [[ARG1]](s8)
; ALL-NEXT:  RET 0, implicit %al

entry:
  store i8 %arg1, i8* @a1_8bit
  store i8 %arg7, i8* @a7_8bit
  store i8 %arg8, i8* @a8_8bit
  ret i8 %arg1
}

@a1_32bit = external global i32
@a7_32bit = external global i32
@a8_32bit = external global i32

define i32 @test_i32_args_8(i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4,
		                        i32 %arg5, i32 %arg6, i32 %arg7, i32 %arg8) {

; ALL-LABEL: name:            test_i32_args_8

; X64: fixedStack:
; X64:  id: [[STACK8:[0-9]+]], type: default, offset: 8, size: 4, alignment: 8,
; X64-NEXT: isImmutable: true,
; X64:  id: [[STACK0:[0-9]+]], type: default, offset: 0, size: 4, alignment: 16,
; X64-NEXT: isImmutable: true,
; X64: liveins: %ecx, %edi, %edx, %esi, %r8d, %r9d
; X64:      [[ARG1:%[0-9]+]](s32) = COPY %edi
; X64-NEXT: %{{[0-9]+}}(s32) = COPY %esi
; X64-NEXT: %{{[0-9]+}}(s32) = COPY %edx
; X64-NEXT: %{{[0-9]+}}(s32) = COPY %ecx
; X64-NEXT: %{{[0-9]+}}(s32) = COPY %r8d
; X64-NEXT: %{{[0-9]+}}(s32) = COPY %r9d
; X64-NEXT: [[ARG7_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK0]]
; X64-NEXT: [[ARG7:%[0-9]+]](s32) = G_LOAD [[ARG7_ADDR]](p0) :: (invariant load 4 from %fixed-stack.[[STACK0]], align 0)
; X64-NEXT: [[ARG8_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK8]]
; X64-NEXT: [[ARG8:%[0-9]+]](s32) = G_LOAD [[ARG8_ADDR]](p0) :: (invariant load 4 from %fixed-stack.[[STACK8]], align 0)

; X32: fixedStack:
; X32:  id: [[STACK28:[0-9]+]], type: default, offset: 28, size: 4, alignment: 4,
; X32-NEXT: isImmutable: true,

; X32:  id: [[STACK24:[0-9]+]], type: default, offset: 24, size: 4, alignment: 8
; X32-NEXT: isImmutable: true,

; X32:  id: [[STACK20:[0-9]+]], type: default, offset: 20, size: 4, alignment: 4
; X32-NEXT: isImmutable: true,

; X32:  id: [[STACK16:[0-9]+]], type: default, offset: 16, size: 4, alignment: 16
; X32-NEXT: isImmutable: true,

; X32:  id: [[STACK12:[0-9]+]], type: default, offset: 12, size: 4, alignment: 4
; X32-NEXT: isImmutable: true,

; X32:  id: [[STACK8:[0-9]+]], type: default, offset: 8, size: 4, alignment: 8
; X32-NEXT: isImmutable: true,

; X32:  id: [[STACK4:[0-9]+]], type: default, offset: 4, size: 4, alignment: 4
; X32-NEXT: isImmutable: true,
; X32:  id: [[STACK0:[0-9]+]], type: default, offset: 0, size: 4, alignment: 16
; X32-NEXT: isImmutable: true,

; X32:       [[ARG1_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK0]]
; X32-NEXT:  [[ARG1:%[0-9]+]](s32) = G_LOAD [[ARG1_ADDR]](p0) :: (invariant load 4 from %fixed-stack.[[STACK0]], align 0)
; X32-NEXT:  [[ARG2_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK4]]
; X32-NEXT:  [[ARG2:%[0-9]+]](s32) = G_LOAD [[ARG2_ADDR]](p0) :: (invariant load 4 from %fixed-stack.[[STACK4]], align 0)
; X32-NEXT:  [[ARG3_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK8]]
; X32-NEXT:  [[ARG3:%[0-9]+]](s32) = G_LOAD [[ARG3_ADDR]](p0) :: (invariant load 4 from %fixed-stack.[[STACK8]], align 0)
; X32-NEXT:  [[ARG4_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK12]]
; X32-NEXT:  [[ARG4:%[0-9]+]](s32) = G_LOAD [[ARG4_ADDR]](p0) :: (invariant load 4 from %fixed-stack.[[STACK12]], align 0)
; X32-NEXT:  [[ARG5_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK16]]
; X32-NEXT:  [[ARG5:%[0-9]+]](s32) = G_LOAD [[ARG5_ADDR]](p0) :: (invariant load 4 from %fixed-stack.[[STACK16]], align 0)
; X32-NEXT:  [[ARG6_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK20]]
; X32-NEXT:  [[ARG6:%[0-9]+]](s32) = G_LOAD [[ARG6_ADDR]](p0) :: (invariant load 4 from %fixed-stack.[[STACK20]], align 0)
; X32-NEXT:  [[ARG7_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK24]]
; X32-NEXT:  [[ARG7:%[0-9]+]](s32) = G_LOAD [[ARG7_ADDR]](p0) :: (invariant load 4 from %fixed-stack.[[STACK24]], align 0)
; X32-NEXT:  [[ARG8_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK28]]
; X32-NEXT:  [[ARG8:%[0-9]+]](s32) = G_LOAD [[ARG8_ADDR]](p0) :: (invariant load 4 from %fixed-stack.[[STACK28]], align 0)

; ALL-NEXT:  [[GADDR_A1:%[0-9]+]](p0) = G_GLOBAL_VALUE @a1_32bit
; ALL-NEXT:  [[GADDR_A7:%[0-9]+]](p0) = G_GLOBAL_VALUE @a7_32bit
; ALL-NEXT:  [[GADDR_A8:%[0-9]+]](p0) = G_GLOBAL_VALUE @a8_32bit
; ALL-NEXT:  G_STORE [[ARG1]](s32), [[GADDR_A1]](p0) :: (store 4 into @a1_32bit)
; ALL-NEXT:  G_STORE [[ARG7]](s32), [[GADDR_A7]](p0) :: (store 4 into @a7_32bit)
; ALL-NEXT:  G_STORE [[ARG8]](s32), [[GADDR_A8]](p0) :: (store 4 into @a8_32bit)
; ALL-NEXT:  %eax = COPY [[ARG1]](s32)
; ALL-NEXT:  RET 0, implicit %eax

entry:
  store i32 %arg1, i32* @a1_32bit
  store i32 %arg7, i32* @a7_32bit
  store i32 %arg8, i32* @a8_32bit
  ret i32 %arg1
}

@a1_64bit = external global i64
@a7_64bit = external global i64
@a8_64bit = external global i64

define i64 @test_i64_args_8(i64 %arg1, i64 %arg2, i64 %arg3, i64 %arg4,
                            i64 %arg5, i64 %arg6, i64 %arg7, i64 %arg8) {

; ALL-LABEL: name:            test_i64_args_8
; X64: fixedStack:
; X64:  id: [[STACK8:[0-9]+]], type: default, offset: 8, size: 8, alignment: 8,
; X64-NEXT: isImmutable: true,
; X64:  id: [[STACK0:[0-9]+]], type: default, offset: 0, size: 8, alignment: 16,
; X64-NEXT: isImmutable: true,
; X64: liveins: %rcx, %rdi, %rdx, %rsi, %r8, %r9
; X64:      [[ARG1:%[0-9]+]](s64) = COPY %rdi
; X64-NEXT: %{{[0-9]+}}(s64) = COPY %rsi
; X64-NEXT: %{{[0-9]+}}(s64) = COPY %rdx
; X64-NEXT: %{{[0-9]+}}(s64) = COPY %rcx
; X64-NEXT: %{{[0-9]+}}(s64) = COPY %r8
; X64-NEXT: %{{[0-9]+}}(s64) = COPY %r9
; X64-NEXT: [[ARG7_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK0]]
; X64-NEXT: [[ARG7:%[0-9]+]](s64) = G_LOAD [[ARG7_ADDR]](p0) :: (invariant load 8 from %fixed-stack.[[STACK0]], align 0)
; X64-NEXT: [[ARG8_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK8]]
; X64-NEXT: [[ARG8:%[0-9]+]](s64) = G_LOAD [[ARG8_ADDR]](p0) :: (invariant load 8 from %fixed-stack.[[STACK8]], align 0)

; X32: fixedStack:
; X32:  id: [[STACK60:[0-9]+]], type: default, offset: 60, size: 4, alignment: 4,
; X32-NEXT: isImmutable: true,
; X32:  id: [[STACK56:[0-9]+]], type: default, offset: 56, size: 4, alignment: 8,
; X32-NEXT: isImmutable: true,
; X32:  id: [[STACK52:[0-9]+]], type: default, offset: 52, size: 4, alignment: 4
; X32-NEXT: isImmutable: true,
; X32:  id: [[STACK48:[0-9]+]], type: default, offset: 48, size: 4, alignment: 16
; X32-NEXT: isImmutable: true,
; X32:  id: [[STACK44:[0-9]+]], type: default, offset: 44, size: 4, alignment: 4
; X32-NEXT: isImmutable: true,
; X32:  id: [[STACK40:[0-9]+]], type: default, offset: 40, size: 4, alignment: 8
; X32-NEXT: isImmutable: true,
; X32:  id: [[STACK36:[0-9]+]], type: default, offset: 36, size: 4, alignment: 4
; X32-NEXT: isImmutable: true,
; X32:  id: [[STACK32:[0-9]+]], type: default, offset: 32, size: 4, alignment: 16
; X32-NEXT: isImmutable: true,
; X32:  id: [[STACK28:[0-9]+]], type: default, offset: 28, size: 4, alignment: 4
; X32-NEXT: isImmutable: true,
; X32:  id: [[STACK24:[0-9]+]], type: default, offset: 24, size: 4, alignment: 8
; X32-NEXT: isImmutable: true,
; X32:  id: [[STACK20:[0-9]+]], type: default, offset: 20, size: 4, alignment: 4
; X32-NEXT: isImmutable: true,
; X32:  id: [[STACK16:[0-9]+]], type: default, offset: 16, size: 4, alignment: 16
; X32-NEXT: isImmutable: true,
; X32:  id: [[STACK12:[0-9]+]], type: default, offset: 12, size: 4, alignment: 4
; X32-NEXT: isImmutable: true,
; X32:  id: [[STACK8:[0-9]+]], type: default, offset: 8, size: 4, alignment: 8
; X32-NEXT: isImmutable: true,
; X32:  id: [[STACK4:[0-9]+]], type: default, offset: 4, size: 4, alignment: 4
; X32-NEXT: isImmutable: true,
; X32:  id: [[STACK0:[0-9]+]], type: default, offset: 0, size: 4, alignment: 16
; X32-NEXT: isImmutable: true,

; X32:      [[ARG1L_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK0]]
; X32-NEXT: [[ARG1L:%[0-9]+]](s32) = G_LOAD [[ARG1L_ADDR]](p0) :: (invariant load 4 from %fixed-stack.[[STACK0]], align 0)
; X32-NEXT: [[ARG1H_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK4]]
; X32-NEXT: [[ARG1H:%[0-9]+]](s32) = G_LOAD [[ARG1H_ADDR]](p0) :: (invariant load 4 from %fixed-stack.[[STACK4]], align 0)
; X32-NEXT: %{{[0-9]+}}(p0) = G_FRAME_INDEX %fixed-stack.[[STACK8]]
; X32-NEXT: %{{[0-9]+}}(s32) = G_LOAD %{{[0-9]+}}(p0) :: (invariant load 4 from %fixed-stack.[[STACK8]], align 0)
; X32-NEXT: %{{[0-9]+}}(p0) = G_FRAME_INDEX %fixed-stack.[[STACK12]]
; X32-NEXT: %{{[0-9]+}}(s32) = G_LOAD %{{[0-9]+}}(p0) :: (invariant load 4 from %fixed-stack.[[STACK12]], align 0)
; X32-NEXT: %{{[0-9]+}}(p0) = G_FRAME_INDEX %fixed-stack.[[STACK16]]
; X32-NEXT: %{{[0-9]+}}(s32) = G_LOAD %{{[0-9]+}}(p0) :: (invariant load 4 from %fixed-stack.[[STACK16]], align 0)
; X32-NEXT: %{{[0-9]+}}(p0) = G_FRAME_INDEX %fixed-stack.[[STACK20]]
; X32-NEXT: %{{[0-9]+}}(s32) = G_LOAD %{{[0-9]+}}(p0) :: (invariant load 4 from %fixed-stack.[[STACK20]], align 0)
; X32-NEXT: %{{[0-9]+}}(p0) = G_FRAME_INDEX %fixed-stack.[[STACK24]]
; X32-NEXT: %{{[0-9]+}}(s32) = G_LOAD %{{[0-9]+}}(p0) :: (invariant load 4 from %fixed-stack.[[STACK24]], align 0)
; X32-NEXT: %{{[0-9]+}}(p0) = G_FRAME_INDEX %fixed-stack.[[STACK28]]
; X32-NEXT: %{{[0-9]+}}(s32) = G_LOAD %{{[0-9]+}}(p0) :: (invariant load 4 from %fixed-stack.[[STACK28]], align 0)
; X32-NEXT: %{{[0-9]+}}(p0) = G_FRAME_INDEX %fixed-stack.[[STACK32]]
; X32-NEXT: %{{[0-9]+}}(s32) = G_LOAD %{{[0-9]+}}(p0) :: (invariant load 4 from %fixed-stack.[[STACK32]], align 0)
; X32-NEXT: %{{[0-9]+}}(p0) = G_FRAME_INDEX %fixed-stack.[[STACK36]]
; X32-NEXT: %{{[0-9]+}}(s32) = G_LOAD %{{[0-9]+}}(p0) :: (invariant load 4 from %fixed-stack.[[STACK36]], align 0)
; X32-NEXT: %{{[0-9]+}}(p0) = G_FRAME_INDEX %fixed-stack.[[STACK40]]
; X32-NEXT: %{{[0-9]+}}(s32) = G_LOAD %{{[0-9]+}}(p0) :: (invariant load 4 from %fixed-stack.[[STACK40]], align 0)
; X32-NEXT: %{{[0-9]+}}(p0) = G_FRAME_INDEX %fixed-stack.[[STACK44]]
; X32-NEXT: %{{[0-9]+}}(s32) = G_LOAD %{{[0-9]+}}(p0) :: (invariant load 4 from %fixed-stack.[[STACK44]], align 0)
; X32-NEXT: [[ARG7L_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK48]]
; X32-NEXT: [[ARG7L:%[0-9]+]](s32) = G_LOAD [[ARG7L_ADDR]](p0) :: (invariant load 4 from %fixed-stack.[[STACK48]], align 0)
; X32-NEXT: [[ARG7H_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK52]]
; X32-NEXT: [[ARG7H:%[0-9]+]](s32) = G_LOAD [[ARG7H_ADDR]](p0) :: (invariant load 4 from %fixed-stack.[[STACK52]], align 0)
; X32-NEXT: [[ARG8L_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK56]]
; X32-NEXT: [[ARG8L:%[0-9]+]](s32) = G_LOAD [[ARG8L_ADDR]](p0) :: (invariant load 4 from %fixed-stack.[[STACK56]], align 0)
; X32-NEXT: [[ARG8H_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK60]]
; X32-NEXT: [[ARG8H:%[0-9]+]](s32) = G_LOAD [[ARG8H_ADDR]](p0) :: (invariant load 4 from %fixed-stack.[[STACK60]], align 0)

; X32-NEXT: [[ARG1:%[0-9]+]](s64) = G_MERGE_VALUES [[ARG1L]](s32), [[ARG1H]](s32)
; ... a bunch more that we don't track ...
; X32-NEXT: G_MERGE_VALUES
; X32-NEXT: G_MERGE_VALUES
; X32-NEXT: G_MERGE_VALUES
; X32-NEXT: G_MERGE_VALUES
; X32-NEXT: G_MERGE_VALUES
; X32-NEXT: [[ARG7:%[0-9]+]](s64) = G_MERGE_VALUES [[ARG7L]](s32), [[ARG7H]](s32)
; X32-NEXT: [[ARG8:%[0-9]+]](s64) = G_MERGE_VALUES [[ARG8L]](s32), [[ARG8H]](s32)

; ALL-NEXT: [[GADDR_A1:%[0-9]+]](p0) = G_GLOBAL_VALUE @a1_64bit
; ALL-NEXT: [[GADDR_A7:%[0-9]+]](p0) = G_GLOBAL_VALUE @a7_64bit
; ALL-NEXT: [[GADDR_A8:%[0-9]+]](p0) = G_GLOBAL_VALUE @a8_64bit
; ALL-NEXT: G_STORE [[ARG1]](s64), [[GADDR_A1]](p0) :: (store 8 into @a1_64bit
; ALL-NEXT: G_STORE [[ARG7]](s64), [[GADDR_A7]](p0) :: (store 8 into @a7_64bit
; ALL-NEXT: G_STORE [[ARG8]](s64), [[GADDR_A8]](p0) :: (store 8 into @a8_64bit

; X64-NEXT: %rax = COPY [[ARG1]](s64)
; X64-NEXT: RET 0, implicit %rax

; X32-NEXT: [[RETL:%[0-9]+]](s32), [[RETH:%[0-9]+]](s32) = G_UNMERGE_VALUES [[ARG1:%[0-9]+]](s64)
; X32-NEXT: %eax = COPY [[RETL:%[0-9]+]](s32)
; X32-NEXT: %edx = COPY [[RETH:%[0-9]+]](s32)
; X32-NEXT: RET 0, implicit %eax, implicit %edx

entry:
  store i64 %arg1, i64* @a1_64bit
  store i64 %arg7, i64* @a7_64bit
  store i64 %arg8, i64* @a8_64bit
  ret i64 %arg1
}

define float @test_float_args(float %arg1, float %arg2) {
; ALL-LABEL:name:            test_float_args

; X64: liveins: %xmm0, %xmm1
; X64:      [[ARG1:%[0-9]+]](s32) = COPY %xmm0
; X64-NEXT: [[ARG2:%[0-9]+]](s32) = COPY %xmm1
; X64-NEXT: %xmm0 = COPY [[ARG2:%[0-9]+]](s32)
; X64-NEXT: RET 0, implicit %xmm0

; X32: fixedStack:
; X32:  id: [[STACK4:[0-9]+]], type: default, offset: 4, size: 4, alignment: 4,
; X32-NEXT: isImmutable: true,
; X32:  id: [[STACK0:[0-9]+]], type: default, offset: 0, size: 4, alignment: 16
; X32-NEXT: isImmutable: true,
; X32:       [[ARG1_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK0]]
; X32-NEXT:  [[ARG1:%[0-9]+]](s32) = G_LOAD [[ARG1_ADDR:%[0-9]+]](p0) :: (invariant load 4 from %fixed-stack.[[STACK0]], align 0)
; X32-NEXT:  [[ARG2_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK4]]
; X32-NEXT:  [[ARG2:%[0-9]+]](s32) = G_LOAD [[ARG2_ADDR:%[0-9]+]](p0) :: (invariant load 4 from %fixed-stack.[[STACK4]], align 0)
; X32-NEXT:  %fp0 = COPY [[ARG2:%[0-9]+]](s32)
; X32-NEXT:  RET 0, implicit %fp0

  ret float %arg2
}

define double @test_double_args(double %arg1, double %arg2) {
; ALL-LABEL:name:            test_double_args
; X64: liveins: %xmm0, %xmm1
; X64:     [[ARG1:%[0-9]+]](s64) = COPY %xmm0
; X64-NEXT: [[ARG2:%[0-9]+]](s64) = COPY %xmm1
; X64-NEXT: %xmm0 = COPY [[ARG2:%[0-9]+]](s64)
; X64-NEXT: RET 0, implicit %xmm0

; X32: fixedStack:
; X32:  id: [[STACK4:[0-9]+]], type: default, offset: 8, size: 8, alignment: 8,
; X32-NEXT: isImmutable: true,

; X32:  id: [[STACK0:[0-9]+]], type: default, offset: 0, size: 8, alignment: 16,
; X32-NEXT: isImmutable: true,

; X32:       [[ARG1_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK0]]
; X32-NEXT:  [[ARG1:%[0-9]+]](s64) = G_LOAD [[ARG1_ADDR:%[0-9]+]](p0) :: (invariant load 8 from %fixed-stack.[[STACK0]], align 0)
; X32-NEXT:  [[ARG2_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK4]]
; X32-NEXT:  [[ARG2:%[0-9]+]](s64) = G_LOAD [[ARG2_ADDR:%[0-9]+]](p0) :: (invariant load 8 from %fixed-stack.[[STACK4]], align 0)
; X32-NEXT:  %fp0 = COPY [[ARG2:%[0-9]+]](s64)
; X32-NEXT:  RET 0, implicit %fp0

  ret double %arg2
}

define <4 x i32> @test_v4i32_args(<4 x i32> %arg1, <4 x i32> %arg2) {
; ALL: name:            test_v4i32_args
; ALL: liveins: %xmm0, %xmm1
; ALL:      [[ARG1:%[0-9]+]](<4 x s32>) = COPY %xmm0
; ALL-NEXT: [[ARG2:%[0-9]+]](<4 x s32>) = COPY %xmm1
; ALL-NEXT: %xmm0 = COPY [[ARG2:%[0-9]+]](<4 x s32>)
; ALL-NEXT: RET 0, implicit %xmm0
  ret <4 x i32> %arg2
}

define <8 x i32> @test_v8i32_args(<8 x i32> %arg1) {
; ALL: name:            test_v8i32_args
; ALL: liveins: %xmm0, %xmm1
; ALL:      [[ARG1L:%[0-9]+]](<4 x s32>) = COPY %xmm0
; ALL-NEXT: [[ARG1H:%[0-9]+]](<4 x s32>) = COPY %xmm1
; ALL-NEXT: [[ARG1:%[0-9]+]](<8 x s32>) = G_MERGE_VALUES [[ARG1L]](<4 x s32>), [[ARG1H]](<4 x s32>)
; ALL-NEXT: [[RETL:%[0-9]+]](<4 x s32>), [[RETH:%[0-9]+]](<4 x s32>) = G_UNMERGE_VALUES [[ARG1:%[0-9]+]](<8 x s32>)
; ALL-NEXT: %xmm0 = COPY [[RETL:%[0-9]+]](<4 x s32>)
; ALL-NEXT: %xmm1 = COPY [[RETH:%[0-9]+]](<4 x s32>)
; ALL-NEXT: RET 0, implicit %xmm0, implicit %xmm1

  ret <8 x i32> %arg1
}

define void @test_void_return() {
; ALL-LABEL: name:            test_void_return
; ALL:        bb.1.entry:
; ALL-NEXT:     RET 0
entry:
  ret void
}

define i32 * @test_memop_i32(i32 * %p1) {
; ALL-LABEL:name:            test_memop_i32
;X64    liveins: %rdi
;X64:       %0(p0) = COPY %rdi
;X64-NEXT:  %rax = COPY %0(p0)
;X64-NEXT:  RET 0, implicit %rax

;X32: fixedStack:
;X32:  id: [[STACK0:[0-9]+]], type: default, offset: 0, size: 4, alignment: 16,
;X32-NEXT: isImmutable: true,
;X32:         %1(p0) = G_FRAME_INDEX %fixed-stack.[[STACK0]]
;X32-NEXT:    %0(p0) = G_LOAD %1(p0) :: (invariant load 4 from %fixed-stack.[[STACK0]], align 0)
;X32-NEXT:    %eax = COPY %0(p0)
;X32-NEXT:    RET 0, implicit %eax

  ret i32 * %p1;
}

declare void @trivial_callee()
define void @test_trivial_call() {
; ALL-LABEL: name:            test_trivial_call

; X32:      ADJCALLSTACKDOWN32 0, 0, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT: CALLpcrel32 @trivial_callee, csr_32, implicit %esp
; X32-NEXT: ADJCALLSTACKUP32 0, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT: RET 0

; X64:      ADJCALLSTACKDOWN64 0, 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp
; X64-NEXT: CALL64pcrel32 @trivial_callee, csr_64, implicit %rsp
; X64-NEXT: ADJCALLSTACKUP64 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp
; X64-NEXT: RET 0

  call void @trivial_callee()
  ret void
}

declare void @simple_arg_callee(i32 %in0, i32 %in1)
define void @test_simple_arg(i32 %in0, i32 %in1) {
; ALL-LABEL: name:            test_simple_arg

; X32:      fixedStack:      
; X32:   - { id: 0, type: default, offset: 4, size: 4, alignment: 4,
; X32-NEXT:  isImmutable: true,
; X32:   - { id: 1, type: default, offset: 0, size: 4, alignment: 16,
; X32-NEXT:  isImmutable: true,
; X32:      body:             |
; X32-NEXT:   bb.1 (%ir-block.0):
; X32-NEXT:     %2(p0) = G_FRAME_INDEX %fixed-stack.1
; X32-NEXT:     %0(s32) = G_LOAD %2(p0) :: (invariant load 4 from %fixed-stack.1, align 0)
; X32-NEXT:     %3(p0) = G_FRAME_INDEX %fixed-stack.0
; X32-NEXT:     %1(s32) = G_LOAD %3(p0) :: (invariant load 4 from %fixed-stack.0, align 0)
; X32-NEXT:     ADJCALLSTACKDOWN32 8, 0, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT:     %4(p0) = COPY %esp
; X32-NEXT:     %5(s32) = G_CONSTANT i32 0
; X32-NEXT:     %6(p0) = G_GEP %4, %5(s32)
; X32-NEXT:     G_STORE %1(s32), %6(p0) :: (store 4 into stack, align 0)
; X32-NEXT:     %7(p0) = COPY %esp
; X32-NEXT:     %8(s32) = G_CONSTANT i32 4
; X32-NEXT:     %9(p0) = G_GEP %7, %8(s32)
; X32-NEXT:     G_STORE %0(s32), %9(p0) :: (store 4 into stack + 4, align 0)
; X32-NEXT:     CALLpcrel32 @simple_arg_callee, csr_32, implicit %esp
; X32-NEXT:     ADJCALLSTACKUP32 8, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT:     RET 0

; X64:      %0(s32) = COPY %edi
; X64-NEXT: %1(s32) = COPY %esi
; X64-NEXT: ADJCALLSTACKDOWN64 0, 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp
; X64-NEXT: %edi = COPY %1(s32)
; X64-NEXT: %esi = COPY %0(s32)
; X64-NEXT: CALL64pcrel32 @simple_arg_callee, csr_64, implicit %rsp, implicit %edi, implicit %esi
; X64-NEXT: ADJCALLSTACKUP64 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp
; X64-NEXT: RET 0

  call void @simple_arg_callee(i32 %in1, i32 %in0)
  ret void
}

declare void @simple_arg8_callee(i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, i32 %arg5, i32 %arg6, i32 %arg7, i32 %arg8)
define void @test_simple_arg8_call(i32 %in0) {
; ALL-LABEL: name:            test_simple_arg8_call

; X32:      fixedStack:      
; X32:   - { id: 0, type: default, offset: 0, size: 4, alignment: 16,
; X32-NEXT:  isImmutable: true,         
; X32:     body:             |
; X32-NEXT:   bb.1 (%ir-block.0):
; X32-NEXT:     %1(p0) = G_FRAME_INDEX %fixed-stack.0
; X32-NEXT:     %0(s32) = G_LOAD %1(p0) :: (invariant load 4 from %fixed-stack.0, align 0)
; X32-NEXT:     ADJCALLSTACKDOWN32 32, 0, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT:     %2(p0) = COPY %esp
; X32-NEXT:     %3(s32) = G_CONSTANT i32 0
; X32-NEXT:     %4(p0) = G_GEP %2, %3(s32)
; X32-NEXT:     G_STORE %0(s32), %4(p0) :: (store 4 into stack, align 0)
; X32-NEXT:     %5(p0) = COPY %esp
; X32-NEXT:     %6(s32) = G_CONSTANT i32 4
; X32-NEXT:     %7(p0) = G_GEP %5, %6(s32)
; X32-NEXT:     G_STORE %0(s32), %7(p0) :: (store 4 into stack + 4, align 0)
; X32-NEXT:     %8(p0) = COPY %esp
; X32-NEXT:     %9(s32) = G_CONSTANT i32 8
; X32-NEXT:     %10(p0) = G_GEP %8, %9(s32)
; X32-NEXT:     G_STORE %0(s32), %10(p0) :: (store 4 into stack + 8, align 0)
; X32-NEXT:     %11(p0) = COPY %esp
; X32-NEXT:     %12(s32) = G_CONSTANT i32 12
; X32-NEXT:     %13(p0) = G_GEP %11, %12(s32)
; X32-NEXT:     G_STORE %0(s32), %13(p0) :: (store 4 into stack + 12, align 0)
; X32-NEXT:     %14(p0) = COPY %esp
; X32-NEXT:     %15(s32) = G_CONSTANT i32 16
; X32-NEXT:     %16(p0) = G_GEP %14, %15(s32)
; X32-NEXT:     G_STORE %0(s32), %16(p0) :: (store 4 into stack + 16, align 0)
; X32-NEXT:     %17(p0) = COPY %esp
; X32-NEXT:     %18(s32) = G_CONSTANT i32 20
; X32-NEXT:     %19(p0) = G_GEP %17, %18(s32)
; X32-NEXT:     G_STORE %0(s32), %19(p0) :: (store 4 into stack + 20, align 0)
; X32-NEXT:     %20(p0) = COPY %esp
; X32-NEXT:     %21(s32) = G_CONSTANT i32 24
; X32-NEXT:     %22(p0) = G_GEP %20, %21(s32)
; X32-NEXT:     G_STORE %0(s32), %22(p0) :: (store 4 into stack + 24, align 0)
; X32-NEXT:     %23(p0) = COPY %esp
; X32-NEXT:     %24(s32) = G_CONSTANT i32 28
; X32-NEXT:     %25(p0) = G_GEP %23, %24(s32)
; X32-NEXT:     G_STORE %0(s32), %25(p0) :: (store 4 into stack + 28, align 0)
; X32-NEXT:     CALLpcrel32 @simple_arg8_callee, csr_32, implicit %esp
; X32-NEXT:     ADJCALLSTACKUP32 32, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT:     RET 0

; X64:          %0(s32) = COPY %edi
; X64-NEXT:     ADJCALLSTACKDOWN64 16, 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp
; X64-NEXT:     %edi = COPY %0(s32)
; X64-NEXT:     %esi = COPY %0(s32)
; X64-NEXT:     %edx = COPY %0(s32)
; X64-NEXT:     %ecx = COPY %0(s32)
; X64-NEXT:     %r8d = COPY %0(s32)
; X64-NEXT:     %r9d = COPY %0(s32)
; X64-NEXT:     %1(p0) = COPY %rsp
; X64-NEXT:     %2(s64) = G_CONSTANT i64 0
; X64-NEXT:     %3(p0) = G_GEP %1, %2(s64)
; X64-NEXT:     G_STORE %0(s32), %3(p0) :: (store 4 into stack, align 0)
; X64-NEXT:     %4(p0) = COPY %rsp
; X64-NEXT:     %5(s64) = G_CONSTANT i64 8
; X64-NEXT:     %6(p0) = G_GEP %4, %5(s64)
; X64-NEXT:     G_STORE %0(s32), %6(p0) :: (store 4 into stack + 8, align 0)
; X64-NEXT:     CALL64pcrel32 @simple_arg8_callee, csr_64, implicit %rsp, implicit %edi, implicit %esi, implicit %edx, implicit %ecx, implicit %r8d, implicit %r9d
; X64-NEXT:     ADJCALLSTACKUP64 16, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp
; X64-NEXT:     RET 0

  call void @simple_arg8_callee(i32 %in0, i32 %in0, i32 %in0, i32 %in0,i32 %in0, i32 %in0, i32 %in0, i32 %in0)
  ret void
}

declare i32 @simple_return_callee(i32 %in0)
define i32 @test_simple_return_callee() {
; ALL-LABEL: name:            test_simple_return_callee

; X32:      %1(s32) = G_CONSTANT i32 5
; X32-NEXT: ADJCALLSTACKDOWN32 4, 0, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT: %2(p0) = COPY %esp
; X32-NEXT: %3(s32) = G_CONSTANT i32 0
; X32-NEXT: %4(p0) = G_GEP %2, %3(s32)
; X32-NEXT: G_STORE %1(s32), %4(p0) :: (store 4 into stack, align 0)
; X32-NEXT: CALLpcrel32 @simple_return_callee, csr_32, implicit %esp, implicit-def %eax
; X32-NEXT: %0(s32) = COPY %eax
; X32-NEXT: ADJCALLSTACKUP32 4, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT: %5(s32) = G_ADD %0, %0
; X32-NEXT: %eax = COPY %5(s32)
; X32-NEXT: RET 0, implicit %eax

; X64:      %1(s32) = G_CONSTANT i32 5                                                                 
; X64-NEXT: ADJCALLSTACKDOWN64 0, 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp          
; X64-NEXT: %edi = COPY %1(s32)                                                                         
; X64-NEXT: CALL64pcrel32 @simple_return_callee, csr_64, implicit %rsp, implicit %edi, implicit-def %eax
; X64-NEXT: %0(s32) = COPY %eax                                                                         
; X64-NEXT: ADJCALLSTACKUP64 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp               
; X64-NEXT: %2(s32) = G_ADD %0, %0                                                                      
; X64-NEXT: %eax = COPY %2(s32)                                                                         
; X64-NEXT: RET 0, implicit %eax
                    
  %call = call i32 @simple_return_callee(i32 5)
  %r = add i32 %call, %call
  ret i32 %r
}

declare <8 x i32> @split_return_callee(<8 x i32> %in0)
define <8 x i32> @test_split_return_callee(<8 x i32> %arg1, <8 x i32> %arg2) {
; ALL-LABEL: name: test_split_return_callee

; X32:       fixedStack:                                                                                                                                                                                   
; X32-NEXT:   - { id: 0, type: default, offset: 0, size: 16, alignment: 16,
; X32-NEXT:       isImmutable: true,                                                                                                            
; X32:       %2(<4 x s32>) = COPY %xmm0                                                                                                                                                                
; X32-NEXT:  %3(<4 x s32>) = COPY %xmm1                                                                                                                                                                
; X32-NEXT:  %4(<4 x s32>) = COPY %xmm2                                                                                                                                                                
; X32-NEXT:  %6(p0) = G_FRAME_INDEX %fixed-stack.0
; X32-NEXT:  %5(<4 x s32>) = G_LOAD %6(p0) :: (invariant load 16 from %fixed-stack.0, align 0)
; X32-NEXT:  %0(<8 x s32>) = G_MERGE_VALUES %2(<4 x s32>), %3(<4 x s32>)
; X32-NEXT:  %1(<8 x s32>) = G_MERGE_VALUES %4(<4 x s32>), %5(<4 x s32>)
; X32-NEXT:  ADJCALLSTACKDOWN32 0, 0, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT:  %8(<4 x s32>), %9(<4 x s32>) = G_UNMERGE_VALUES %1(<8 x s32>)
; X32-NEXT:  %xmm0 = COPY %8(<4 x s32>)
; X32-NEXT:  %xmm1 = COPY %9(<4 x s32>)
; X32-NEXT:  CALLpcrel32 @split_return_callee, csr_32, implicit %esp, implicit %xmm0, implicit %xmm1, implicit-def %xmm0, implicit-def %xmm1
; X32-NEXT:  %10(<4 x s32>) = COPY %xmm0
; X32-NEXT:  %11(<4 x s32>) = COPY %xmm1
; X32-NEXT:  %7(<8 x s32>) = G_MERGE_VALUES %10(<4 x s32>), %11(<4 x s32>)
; X32-NEXT:  ADJCALLSTACKUP32 0, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT:  %12(<8 x s32>) = G_ADD %0, %7
; X32-NEXT:  %13(<4 x s32>), %14(<4 x s32>) = G_UNMERGE_VALUES %12(<8 x s32>)
; X32-NEXT:  %xmm0 = COPY %13(<4 x s32>)
; X32-NEXT:  %xmm1 = COPY %14(<4 x s32>)
; X32-NEXT:  RET 0, implicit %xmm0, implicit %xmm1    

; X64:       %2(<4 x s32>) = COPY %xmm0
; X64-NEXT:  %3(<4 x s32>) = COPY %xmm1
; X64-NEXT:  %4(<4 x s32>) = COPY %xmm2
; X64-NEXT:  %5(<4 x s32>) = COPY %xmm3
; X64-NEXT:  %0(<8 x s32>) = G_MERGE_VALUES %2(<4 x s32>), %3(<4 x s32>)
; X64-NEXT:  %1(<8 x s32>) = G_MERGE_VALUES %4(<4 x s32>), %5(<4 x s32>)
; X64-NEXT:  ADJCALLSTACKDOWN64 0, 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp
; X64-NEXT:  %7(<4 x s32>), %8(<4 x s32>) = G_UNMERGE_VALUES %1(<8 x s32>)
; X64-NEXT:  %xmm0 = COPY %7(<4 x s32>)
; X64-NEXT:  %xmm1 = COPY %8(<4 x s32>)
; X64-NEXT:  CALL64pcrel32 @split_return_callee, csr_64, implicit %rsp, implicit %xmm0, implicit %xmm1, implicit-def %xmm0, implicit-def %xmm1
; X64-NEXT:  %9(<4 x s32>) = COPY %xmm0
; X64-NEXT:  %10(<4 x s32>) = COPY %xmm1
; X64-NEXT:  %6(<8 x s32>) = G_MERGE_VALUES %9(<4 x s32>), %10(<4 x s32>)
; X64-NEXT:  ADJCALLSTACKUP64 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp
; X64-NEXT:  %11(<8 x s32>) = G_ADD %0, %6
; X64-NEXT:  %12(<4 x s32>), %13(<4 x s32>) = G_UNMERGE_VALUES %11(<8 x s32>)
; X64-NEXT:  %xmm0 = COPY %12(<4 x s32>)
; X64-NEXT:  %xmm1 = COPY %13(<4 x s32>)
; X64-NEXT:  RET 0, implicit %xmm0, implicit %xmm1    
  
  %call = call <8 x i32> @split_return_callee(<8 x i32> %arg2)
  %r = add <8 x i32> %arg1, %call
  ret  <8 x i32> %r
}

define void @test_indirect_call(void()* %func) {
; ALL-LABEL: name:            test_indirect_call

; X32:       registers:       
; X32-NEXT:   - { id: 0, class: gr32, preferred-register: '' }
; X32-NEXT:   - { id: 1, class: _, preferred-register: '' }
; X32:       %1(p0) = G_FRAME_INDEX %fixed-stack.0
; X32-NEXT:  %0(p0) = G_LOAD %1(p0) :: (invariant load 4 from %fixed-stack.0, align 0)
; X32-NEXT:  ADJCALLSTACKDOWN32 0, 0, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT:  CALL32r %0(p0), csr_32, implicit %esp
; X32-NEXT:  ADJCALLSTACKUP32 0, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT:  RET 0

; X64:      registers:       
; X64-NEXT:    - { id: 0, class: gr64, preferred-register: '' }
; X64:       %0(p0) = COPY %rdi
; X64-NEXT:  ADJCALLSTACKDOWN64 0, 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp
; X64-NEXT:  CALL64r %0(p0), csr_64, implicit %rsp
; X64-NEXT:  ADJCALLSTACKUP64 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp
; X64-NEXT:  RET 0

  call void %func()
  ret void
}


declare void @take_char(i8)
define void @test_abi_exts_call(i8* %addr) {
; ALL-LABEL: name:            test_abi_exts_call

; X32:       fixedStack:      
; X32-NEXT:   - { id: 0, type: default, offset: 0, size: 4, alignment: 16, 
; X32-NEXT:       isImmutable: true, 
; X32:       %1(p0) = G_FRAME_INDEX %fixed-stack.0
; X32-NEXT:  %0(p0) = G_LOAD %1(p0) :: (invariant load 4 from %fixed-stack.0, align 0)
; X32-NEXT:  %2(s8) = G_LOAD %0(p0) :: (load 1 from %ir.addr)
; X32-NEXT:  ADJCALLSTACKDOWN32 4, 0, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT:  %3(p0) = COPY %esp
; X32-NEXT:  %4(s32) = G_CONSTANT i32 0
; X32-NEXT:  %5(p0) = G_GEP %3, %4(s32)
; X32-NEXT:  G_STORE %2(s8), %5(p0) :: (store 4 into stack, align 0)
; X32-NEXT:  CALLpcrel32 @take_char, csr_32, implicit %esp
; X32-NEXT:  ADJCALLSTACKUP32 4, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT:  ADJCALLSTACKDOWN32 4, 0, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT:  %6(p0) = COPY %esp
; X32-NEXT:  %7(s32) = G_CONSTANT i32 0
; X32-NEXT:  %8(p0) = G_GEP %6, %7(s32)
; X32-NEXT:  %9(s32) = G_SEXT %2(s8)
; X32-NEXT:  G_STORE %9(s32), %8(p0) :: (store 4 into stack, align 0)
; X32-NEXT:  CALLpcrel32 @take_char, csr_32, implicit %esp
; X32-NEXT:  ADJCALLSTACKUP32 4, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT:  ADJCALLSTACKDOWN32 4, 0, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT:  %10(p0) = COPY %esp
; X32-NEXT:  %11(s32) = G_CONSTANT i32 0
; X32-NEXT:  %12(p0) = G_GEP %10, %11(s32)
; X32-NEXT:  %13(s32) = G_ZEXT %2(s8)
; X32-NEXT:  G_STORE %13(s32), %12(p0) :: (store 4 into stack, align 0)
; X32-NEXT:  CALLpcrel32 @take_char, csr_32, implicit %esp
; X32-NEXT:  ADJCALLSTACKUP32 4, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT:  RET 0

; X64:       %0(p0) = COPY %rdi
; X64-NEXT:  %1(s8) = G_LOAD %0(p0) :: (load 1 from %ir.addr)
; X64-NEXT:  ADJCALLSTACKDOWN64 0, 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp
; X64-NEXT:  %edi = COPY %1(s8)
; X64-NEXT:  CALL64pcrel32 @take_char, csr_64, implicit %rsp, implicit %edi
; X64-NEXT:  ADJCALLSTACKUP64 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp
; X64-NEXT:  ADJCALLSTACKDOWN64 0, 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp
; X64-NEXT:  %2(s32) = G_SEXT %1(s8)
; X64-NEXT:  %edi = COPY %2(s32)
; X64-NEXT:  CALL64pcrel32 @take_char, csr_64, implicit %rsp, implicit %edi
; X64-NEXT:  ADJCALLSTACKUP64 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp
; X64-NEXT:  ADJCALLSTACKDOWN64 0, 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp
; X64-NEXT:  %3(s32) = G_ZEXT %1(s8)
; X64-NEXT:  %edi = COPY %3(s32)
; X64-NEXT:  CALL64pcrel32 @take_char, csr_64, implicit %rsp, implicit %edi
; X64-NEXT:  ADJCALLSTACKUP64 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp
; X64-NEXT:  RET 0

  %val = load i8, i8* %addr
  call void @take_char(i8 %val)
  call void @take_char(i8 signext %val)
  call void @take_char(i8 zeroext %val)
  ret void
}

declare void @variadic_callee(i8*, ...)
define void @test_variadic_call_1(i8** %addr_ptr, i32* %val_ptr) {
; ALL-LABEL: name:            test_variadic_call_1

; X32:      fixedStack:      
; X32-NEXT:  - { id: 0, type: default, offset: 4, size: 4, alignment: 4, stack-id: 0, 
; X32-NEXT:      isImmutable: true, isAliased: false, callee-saved-register: '' }
; X32-NEXT:  - { id: 1, type: default, offset: 0, size: 4, alignment: 16, stack-id: 0, 
; X32-NEXT:      isImmutable: true, isAliased: false, callee-saved-register: '' }
; X32:         %2(p0) = G_FRAME_INDEX %fixed-stack.1
; X32-NEXT:    %0(p0) = G_LOAD %2(p0) :: (invariant load 4 from %fixed-stack.1, align 0)
; X32-NEXT:    %3(p0) = G_FRAME_INDEX %fixed-stack.0
; X32-NEXT:    %1(p0) = G_LOAD %3(p0) :: (invariant load 4 from %fixed-stack.0, align 0)
; X32-NEXT:    %4(p0) = G_LOAD %0(p0) :: (load 4 from %ir.addr_ptr)
; X32-NEXT:    %5(s32) = G_LOAD %1(p0) :: (load 4 from %ir.val_ptr)
; X32-NEXT:    ADJCALLSTACKDOWN32 8, 0, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT:    %6(p0) = COPY %esp
; X32-NEXT:    %7(s32) = G_CONSTANT i32 0
; X32-NEXT:    %8(p0) = G_GEP %6, %7(s32)
; X32-NEXT:    G_STORE %4(p0), %8(p0) :: (store 4 into stack, align 0)
; X32-NEXT:    %9(p0) = COPY %esp
; X32-NEXT:    %10(s32) = G_CONSTANT i32 4
; X32-NEXT:    %11(p0) = G_GEP %9, %10(s32)
; X32-NEXT:    G_STORE %5(s32), %11(p0) :: (store 4 into stack + 4, align 0)
; X32-NEXT:    CALLpcrel32 @variadic_callee, csr_32, implicit %esp
; X32-NEXT:    ADJCALLSTACKUP32 8, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT:    RET 0
  
; X64:         %0(p0) = COPY %rdi
; X64-NEXT:    %1(p0) = COPY %rsi
; X64-NEXT:    %2(p0) = G_LOAD %0(p0) :: (load 8 from %ir.addr_ptr)
; X64-NEXT:    %3(s32) = G_LOAD %1(p0) :: (load 4 from %ir.val_ptr)
; X64-NEXT:    ADJCALLSTACKDOWN64 0, 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp
; X64-NEXT:    %rdi = COPY %2(p0)
; X64-NEXT:    %esi = COPY %3(s32)
; X64-NEXT:    %al = MOV8ri 0
; X64-NEXT:    CALL64pcrel32 @variadic_callee, csr_64, implicit %rsp, implicit %rdi, implicit %esi, implicit %al
; X64-NEXT:    ADJCALLSTACKUP64 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp
; X64-NEXT:    RET 0
  
  %addr = load i8*, i8** %addr_ptr
  %val = load i32, i32* %val_ptr
  call void (i8*, ...) @variadic_callee(i8* %addr, i32 %val)
  ret void
}

define void @test_variadic_call_2(i8** %addr_ptr, double* %val_ptr) {
; ALL-LABEL: name:            test_variadic_call_2

; X32:      fixedStack:      
; X32-NEXT:  - { id: 0, type: default, offset: 4, size: 4, alignment: 4, stack-id: 0, 
; X32-NEXT:      isImmutable: true, isAliased: false, callee-saved-register: '' }
; X32-NEXT:  - { id: 1, type: default, offset: 0, size: 4, alignment: 16, stack-id: 0, 
; X32-NEXT:      isImmutable: true, isAliased: false, callee-saved-register: '' }
; X32:         %2(p0) = G_FRAME_INDEX %fixed-stack.1
; X32-NEXT:    %0(p0) = G_LOAD %2(p0) :: (invariant load 4 from %fixed-stack.1, align 0)
; X32-NEXT:    %3(p0) = G_FRAME_INDEX %fixed-stack.0
; X32-NEXT:    %1(p0) = G_LOAD %3(p0) :: (invariant load 4 from %fixed-stack.0, align 0)
; X32-NEXT:    %4(p0) = G_LOAD %0(p0) :: (load 4 from %ir.addr_ptr)
; X32-NEXT:    %5(s64) = G_LOAD %1(p0) :: (load 8 from %ir.val_ptr, align 4)
; X32-NEXT:    ADJCALLSTACKDOWN32 12, 0, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT:    %6(p0) = COPY %esp
; X32-NEXT:    %7(s32) = G_CONSTANT i32 0
; X32-NEXT:    %8(p0) = G_GEP %6, %7(s32)
; X32-NEXT:    G_STORE %4(p0), %8(p0) :: (store 4 into stack, align 0)
; X32-NEXT:    %9(p0) = COPY %esp
; X32-NEXT:    %10(s32) = G_CONSTANT i32 4
; X32-NEXT:    %11(p0) = G_GEP %9, %10(s32)
; X32-NEXT:    G_STORE %5(s64), %11(p0) :: (store 8 into stack + 4, align 0)
; X32-NEXT:    CALLpcrel32 @variadic_callee, csr_32, implicit %esp
; X32-NEXT:    ADJCALLSTACKUP32 12, 0, implicit-def %esp, implicit-def %eflags, implicit %esp
; X32-NEXT:    RET 0
  
; X64:         %1(p0) = COPY %rsi
; X64-NEXT:    %2(p0) = G_LOAD %0(p0) :: (load 8 from %ir.addr_ptr)
; X64-NEXT:    %3(s64) = G_LOAD %1(p0) :: (load 8 from %ir.val_ptr)
; X64-NEXT:    ADJCALLSTACKDOWN64 0, 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp
; X64-NEXT:    %rdi = COPY %2(p0)
; X64-NEXT:    %xmm0 = COPY %3(s64)
; X64-NEXT:    %al = MOV8ri 1
; X64-NEXT:    CALL64pcrel32 @variadic_callee, csr_64, implicit %rsp, implicit %rdi, implicit %xmm0, implicit %al
; X64-NEXT:    ADJCALLSTACKUP64 0, 0, implicit-def %rsp, implicit-def %eflags, implicit %rsp
; X64-NEXT:    RET 0
  
  %addr = load i8*, i8** %addr_ptr
  %val = load double, double* %val_ptr
  call void (i8*, ...) @variadic_callee(i8* %addr, double %val)
  ret void
}
