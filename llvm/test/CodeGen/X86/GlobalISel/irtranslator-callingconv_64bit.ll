; RUN: llc -mtriple=x86_64-linux-gnu -global-isel -stop-after=irtranslator < %s -o - | FileCheck %s --check-prefix=X64

@a1_64bit = external global i64
@a7_64bit = external global i64
@a8_64bit = external global i64

define void @test_i64_args_8(i64 %arg1, i64 %arg2, i64 %arg3, i64 %arg4, 
		            i64 %arg5, i64 %arg6, i64 %arg7, i64 %arg8) {
; X64-LABEL: name:            test_i64_args_8
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
; X64-NEXT: [[GADDR_A1:%[0-9]+]](p0) = G_GLOBAL_VALUE @a1_64bit
; X64-NEXT: [[GADDR_A7:%[0-9]+]](p0) = G_GLOBAL_VALUE @a7_64bit
; X64-NEXT: [[GADDR_A8:%[0-9]+]](p0) = G_GLOBAL_VALUE @a8_64bit
; X64-NEXT: G_STORE [[ARG1]](s64), [[GADDR_A1]](p0) :: (store 8 into @a1_64bit)
; X64-NEXT: G_STORE [[ARG7]](s64), [[GADDR_A7]](p0) :: (store 8 into @a7_64bit)
; X64-NEXT: G_STORE [[ARG8]](s64), [[GADDR_A8]](p0) :: (store 8 into @a8_64bit)
entry:
  store i64 %arg1, i64* @a1_64bit
  store i64 %arg7, i64* @a7_64bit
  store i64 %arg8, i64* @a8_64bit 
  ret void
}
