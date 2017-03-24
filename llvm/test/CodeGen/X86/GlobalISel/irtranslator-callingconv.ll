; RUN: llc -mtriple=i386-linux-gnu   -global-isel -stop-after=irtranslator < %s -o - | FileCheck %s --check-prefix=ALL --check-prefix=X32
; RUN: llc -mtriple=x86_64-linux-gnu -global-isel -stop-after=irtranslator < %s -o - | FileCheck %s --check-prefix=ALL --check-prefix=X64

@a1_8bit = external global i8
@a7_8bit = external global i8
@a8_8bit = external global i8

define i8 @test_i8_args_8(i8 %arg1, i8 %arg2, i8 %arg3, i8 %arg4,
		                      i8 %arg5, i8 %arg6, i8 %arg7, i8 %arg8) {

; ALL-LABEL: name:            test_i8_args_8

; X64: fixedStack:
; X64:  id: [[STACK8:[0-9]+]], offset: 8, size: 1, alignment: 8, isImmutable: true, isAliased: false
; X64:  id: [[STACK0:[0-9]+]], offset: 0, size: 1, alignment: 16, isImmutable: true, isAliased: false
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
; X32:  id: [[STACK28:[0-9]+]], offset: 28, size: 1, alignment: 4, isImmutable: true, isAliased: false }
; X32:  id: [[STACK24:[0-9]+]], offset: 24, size: 1, alignment: 8, isImmutable: true, isAliased: false }
; X32:  id: [[STACK20:[0-9]+]], offset: 20, size: 1, alignment: 4, isImmutable: true, isAliased: false }
; X32:  id: [[STACK16:[0-9]+]], offset: 16, size: 1, alignment: 16, isImmutable: true, isAliased: false }
; X32:  id: [[STACK12:[0-9]+]], offset: 12, size: 1, alignment: 4, isImmutable: true, isAliased: false }
; X32:  id: [[STACK8:[0-9]+]],  offset: 8, size: 1, alignment: 8, isImmutable: true, isAliased: false }
; X32:  id: [[STACK4:[0-9]+]],  offset: 4, size: 1, alignment: 4, isImmutable: true, isAliased: false }
; X32:  id: [[STACK0:[0-9]+]],  offset: 0, size: 1, alignment: 16, isImmutable: true, isAliased: false }
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
; X64:  id: [[STACK8:[0-9]+]], offset: 8, size: 4, alignment: 8, isImmutable: true, isAliased: false
; X64:  id: [[STACK0:[0-9]+]], offset: 0, size: 4, alignment: 16, isImmutable: true, isAliased: false
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
; X32:  id: [[STACK28:[0-9]+]], offset: 28, size: 4, alignment: 4, isImmutable: true, isAliased: false }
; X32:  id: [[STACK24:[0-9]+]], offset: 24, size: 4, alignment: 8, isImmutable: true, isAliased: false }
; X32:  id: [[STACK20:[0-9]+]], offset: 20, size: 4, alignment: 4, isImmutable: true, isAliased: false }
; X32:  id: [[STACK16:[0-9]+]], offset: 16, size: 4, alignment: 16, isImmutable: true, isAliased: false }
; X32:  id: [[STACK12:[0-9]+]], offset: 12, size: 4, alignment: 4, isImmutable: true, isAliased: false }
; X32:  id: [[STACK8:[0-9]+]],  offset: 8, size: 4, alignment: 8, isImmutable: true, isAliased: false }
; X32:  id: [[STACK4:[0-9]+]],  offset: 4, size: 4, alignment: 4, isImmutable: true, isAliased: false }
; X32:  id: [[STACK0:[0-9]+]],  offset: 0, size: 4, alignment: 16, isImmutable: true, isAliased: false }
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
; X64:  id: [[STACK8:[0-9]+]], offset: 8, size: 8, alignment: 8, isImmutable: true, isAliased: false
; X64:  id: [[STACK0:[0-9]+]], offset: 0, size: 8, alignment: 16, isImmutable: true, isAliased: false
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
; X32:  id: [[STACK60:[0-9]+]], offset: 60, size: 4, alignment: 4, isImmutable: true, isAliased: false }
; X32:  id: [[STACK56:[0-9]+]], offset: 56, size: 4, alignment: 8, isImmutable: true, isAliased: false }
; X32:  id: [[STACK52:[0-9]+]], offset: 52, size: 4, alignment: 4, isImmutable: true, isAliased: false }
; X32:  id: [[STACK48:[0-9]+]], offset: 48, size: 4, alignment: 16, isImmutable: true, isAliased: false }
; X32:  id: [[STACK44:[0-9]+]], offset: 44, size: 4, alignment: 4, isImmutable: true, isAliased: false }
; X32:  id: [[STACK40:[0-9]+]], offset: 40, size: 4, alignment: 8, isImmutable: true, isAliased: false }
; X32:  id: [[STACK36:[0-9]+]], offset: 36, size: 4, alignment: 4, isImmutable: true, isAliased: false }
; X32:  id: [[STACK32:[0-9]+]], offset: 32, size: 4, alignment: 16, isImmutable: true, isAliased: false }
; X32:  id: [[STACK28:[0-9]+]], offset: 28, size: 4, alignment: 4, isImmutable: true, isAliased: false }
; X32:  id: [[STACK24:[0-9]+]], offset: 24, size: 4, alignment: 8, isImmutable: true, isAliased: false }
; X32:  id: [[STACK20:[0-9]+]], offset: 20, size: 4, alignment: 4, isImmutable: true, isAliased: false }
; X32:  id: [[STACK16:[0-9]+]], offset: 16, size: 4, alignment: 16, isImmutable: true, isAliased: false }
; X32:  id: [[STACK12:[0-9]+]], offset: 12, size: 4, alignment: 4, isImmutable: true, isAliased: false }
; X32:  id: [[STACK8:[0-9]+]], offset: 8, size: 4, alignment: 8, isImmutable: true, isAliased: false }
; X32:  id: [[STACK4:[0-9]+]], offset: 4, size: 4, alignment: 4, isImmutable: true, isAliased: false }
; X32:  id: [[STACK0:[0-9]+]], offset: 0, size: 4, alignment: 16, isImmutable: true, isAliased: false }

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

; X32-NEXT: [[UNDEF:%[0-9]+]](s64) = IMPLICIT_DEF
; X32-NEXT: [[ARG1_TMP0:%[0-9]+]](s64) = G_INSERT [[UNDEF]], [[ARG1L]](s32), 0
; X32-NEXT: [[ARG1_TMP1:%[0-9]+]](s64) = G_INSERT [[ARG1_TMP0]], [[ARG1H]](s32), 32
; X32-NEXT: [[ARG1:%[0-9]+]](s64) = COPY [[ARG1_TMP1]]
  ; ... a bunch more that we don't track ...
  ; X32: IMPLICIT_DEF
  ; X32: IMPLICIT_DEF
  ; X32: IMPLICIT_DEF
  ; X32: IMPLICIT_DEF
  ; X32: IMPLICIT_DEF
; X32: [[UNDEF:%[0-9]+]](s64) = IMPLICIT_DEF
; X32-NEXT: [[ARG7_TMP0:%[0-9]+]](s64) = G_INSERT [[UNDEF]], [[ARG7L]](s32), 0
; X32-NEXT: [[ARG7_TMP1:%[0-9]+]](s64) = G_INSERT [[ARG7_TMP0]], [[ARG7H]](s32), 32
; X32-NEXT: [[ARG7:%[0-9]+]](s64) = COPY [[ARG7_TMP1]]
; X32-NEXT: [[UNDEF:%[0-9]+]](s64) = IMPLICIT_DEF
; X32-NEXT: [[ARG8_TMP0:%[0-9]+]](s64) = G_INSERT [[UNDEF]], [[ARG8L]](s32), 0
; X32-NEXT: [[ARG8_TMP1:%[0-9]+]](s64) = G_INSERT [[ARG8_TMP0]], [[ARG8H]](s32), 32
; X32-NEXT: [[ARG8:%[0-9]+]](s64) = COPY [[ARG8_TMP1]]

; ALL-NEXT: [[GADDR_A1:%[0-9]+]](p0) = G_GLOBAL_VALUE @a1_64bit
; ALL-NEXT: [[GADDR_A7:%[0-9]+]](p0) = G_GLOBAL_VALUE @a7_64bit
; ALL-NEXT: [[GADDR_A8:%[0-9]+]](p0) = G_GLOBAL_VALUE @a8_64bit
; ALL-NEXT: G_STORE [[ARG1]](s64), [[GADDR_A1]](p0) :: (store 8 into @a1_64bit
; ALL-NEXT: G_STORE [[ARG7]](s64), [[GADDR_A7]](p0) :: (store 8 into @a7_64bit
; ALL-NEXT: G_STORE [[ARG8]](s64), [[GADDR_A8]](p0) :: (store 8 into @a8_64bit

; X64-NEXT: %rax = COPY [[ARG1]](s64)
; X64-NEXT: RET 0, implicit %rax

; X32-NEXT: [[RETL:%[0-9]+]](s32) = G_EXTRACT [[ARG1:%[0-9]+]](s64), 0
; X32-NEXT: [[RETH:%[0-9]+]](s32) = G_EXTRACT [[ARG1:%[0-9]+]](s64), 32
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
; X32:  id: [[STACK4:[0-9]+]], offset: 4, size: 4, alignment: 4, isImmutable: true, isAliased: false }
; X32:  id: [[STACK0:[0-9]+]], offset: 0, size: 4, alignment: 16, isImmutable: true, isAliased: false }
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
; X32:  id: [[STACK4:[0-9]+]], offset: 8, size: 8, alignment: 8, isImmutable: true, isAliased: false }
; X32:  id: [[STACK0:[0-9]+]], offset: 0, size: 8, alignment: 16, isImmutable: true, isAliased: false }
; X32:       [[ARG1_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK0]]
; X32-NEXT:  [[ARG1:%[0-9]+]](s64) = G_LOAD [[ARG1_ADDR:%[0-9]+]](p0) :: (invariant load 8 from %fixed-stack.[[STACK0]], align 0)
; X32-NEXT:  [[ARG2_ADDR:%[0-9]+]](p0) = G_FRAME_INDEX %fixed-stack.[[STACK4]]
; X32-NEXT:  [[ARG2:%[0-9]+]](s64) = G_LOAD [[ARG2_ADDR:%[0-9]+]](p0) :: (invariant load 8 from %fixed-stack.[[STACK4]], align 0)
; X32-NEXT:  %fp0 = COPY [[ARG2:%[0-9]+]](s64)
; X32-NEXT:  RET 0, implicit %fp0

  ret double %arg2
}

define i32 * @test_memop_i32(i32 * %p1) {
; ALL-LABEL:name:            test_memop_i32
;X64    liveins: %rdi
;X64:       %0(p0) = COPY %rdi
;X64-NEXT:  %rax = COPY %0(p0)
;X64-NEXT:  RET 0, implicit %rax

;X32: fixedStack:
;X32:  id: [[STACK0:[0-9]+]], offset: 0, size: 4, alignment: 16, isImmutable: true, isAliased: false }
;X32:         %1(p0) = G_FRAME_INDEX %fixed-stack.[[STACK0]]
;X32-NEXT:    %0(p0) = G_LOAD %1(p0) :: (invariant load 4 from %fixed-stack.[[STACK0]], align 0)
;X32-NEXT:    %eax = COPY %0(p0)
;X32-NEXT:    RET 0, implicit %eax

  ret i32 * %p1;
}