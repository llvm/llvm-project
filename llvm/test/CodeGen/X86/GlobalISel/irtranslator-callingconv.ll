; RUN: llc -mtriple=i386-linux-gnu   -global-isel -stop-after=irtranslator < %s -o - | FileCheck %s --check-prefix=ALL --check-prefix=X32
; RUN: llc -mtriple=x86_64-linux-gnu -global-isel -stop-after=irtranslator < %s -o - | FileCheck %s --check-prefix=ALL --check-prefix=X64

@a1_8bit = external global i8
@a7_8bit = external global i8
@a8_8bit = external global i8

define void @test_i8_args_8(i8 %arg1, i8 %arg2, i8 %arg3, i8 %arg4, 
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
entry:
  store i8 %arg1, i8* @a1_8bit
  store i8 %arg7, i8* @a7_8bit
  store i8 %arg8, i8* @a8_8bit
  ret void
}

@a1_32bit = external global i32
@a7_32bit = external global i32
@a8_32bit = external global i32

define void @test_i32_args_8(i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, 
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
entry:
  store i32 %arg1, i32* @a1_32bit
  store i32 %arg7, i32* @a7_32bit
  store i32 %arg8, i32* @a8_32bit 
  ret void
}
