; RUN: llc -mtriple=thumbv7m-arm-none-eabi -mcpu=cortex-m3 < %s | FileCheck %s

%struct.mystruct = type { [31 x i8] }

@.str = private unnamed_addr constant [31 x i8] c"012345678901234567890123456789\00", align 1
@.str.1 = private unnamed_addr constant [21 x i8] c"01234567890123456789\00", align 1

@myglobal = common global %struct.mystruct zeroinitializer, align 1

define void @foo() #0 {
entry:
; CHECK-LABEL: foo:
; CHECK:      __aeabi_memcpy
; CHECK-NOT:  ldm
  %mystring = alloca [31 x i8], align 1
  %0 = getelementptr inbounds [31 x i8], [31 x i8]* %mystring, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %0, i8* getelementptr inbounds ([31 x i8], [31 x i8]* @.str, i32 0, i32 0), i32 31, i32 1, i1 false)
  ret void
}

define void @bar() #0 {
entry:
; CHECK-LABEL: bar:
; CHECK-NOT:   __aeabi_memcpy
  %mystring = alloca [31 x i8], align 1
  %0 = getelementptr inbounds [31 x i8], [31 x i8]* %mystring, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %0, i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.1, i32 0, i32 0), i32 21, i32 1, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i32, i1) #1

attributes #0 = { minsize noinline nounwind optsize }
