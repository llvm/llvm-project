; RUN: opt < %s "-default-data-layout=e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32" -simplify-libcalls -S | FileCheck %s
@.str1 = private constant [11 x i8] c"(){};[]&|:\00", align 4

; check that simplify libcalls will not replace a call with one calling
; convention with a new call with a different calling convention.

; CHECK: define arm_aapcscc i32 @foo(i32 %argc)
; CHECK: call arm_aapcscc  i8* @strchr
define arm_aapcscc i32 @foo(i32 %argc) nounwind {
bb.nph:
  call arm_aapcscc  i8* @strchr(i8* getelementptr ([11 x i8]* @.str1, i32 0,
i32 0), i32 %argc) nounwind readonly
  ret i32 0
}

declare arm_aapcscc i8* @strchr(i8*, i32) nounwind readonly
