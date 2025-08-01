; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_52 -mattr=+ptx64 -O0 | FileCheck %s --check-prefixes=SM_52,COMMON
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx64 -O0 | FileCheck %s --check-prefixes=SM_70,COMMON
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx72 -O0 | FileCheck %s --check-prefixes=SM_90,COMMON

@.str = private unnamed_addr constant [12 x i8] c"__CUDA_ARCH\00"
@.str1 = constant [11 x i8] c"__CUDA_FTZ\00"

declare i32 @__nvvm_reflect(ptr)

; COMMON-LABEL: .visible .func  (.param .b32 func_retval0) foo()
; SM_52: st.param.b32    [func_retval0], 3;
; SM_70: st.param.b32    [func_retval0], 2;
; SM_90: st.param.b32    [func_retval0], 1;
; COMMON-NEXT: ret;

define i32 @foo() {
entry:
  %call = call i32 @__nvvm_reflect(ptr @.str)
  %cmp = icmp uge i32 %call, 900
  br i1 %cmp, label %if.then, label %if.else

if.then:
  br label %return

if.else:
  %call1 = call i32 @__nvvm_reflect(ptr @.str)
  %cmp2 = icmp uge i32 %call1, 700
  br i1 %cmp2, label %if.then3, label %if.else4

if.then3:
  br label %return

if.else4:
  %call5 = call i32 @__nvvm_reflect(ptr @.str)
  %cmp6 = icmp uge i32 %call5, 520
  br i1 %cmp6, label %if.then7, label %if.else8

if.then7:
  br label %return

if.else8:
  br label %return

return:
  %retval.0 = phi i32 [ 1, %if.then ], [ 2, %if.then3 ], [ 3, %if.then7 ], [ 4, %if.else8 ]
  ret i32 %retval.0
}

; COMMON-LABEL: .visible .func  (.param .b32 func_retval0) bar()
; SM_52: st.param.b32    [func_retval0], 2;
; SM_70: st.param.b32    [func_retval0], 1;
; SM_90: st.param.b32    [func_retval0], 1;
; COMMON-NEXT: ret;
define i32 @bar() {
entry:
  %call = call i32 @__nvvm_reflect(ptr @.str)
  %cmp = icmp uge i32 %call, 700
  br i1 %cmp, label %if.then, label %if.else

if.then:
  br label %if.end

if.else:
  br label %if.end

if.end:
  %x = phi i32 [ 1, %if.then ], [ 2, %if.else ]
  ret i32 %x
}

; SM_52-NOT: valid;
; SM_70: valid;
; SM_90: valid;
define void @baz() {
entry:
  %call = call i32 @__nvvm_reflect(ptr @.str)
  %cmp = icmp uge i32 %call, 700
  br i1 %cmp, label %if.then, label %if.end

if.then:
  call void asm sideeffect "valid;\0A", ""()
  br label %if.end

if.end:
  ret void
}

; SM_52: .visible .func  (.param .b32 func_retval0) qux()
; SM_52: st.param.b32    [func_retval0], 3;
; SM_70: st.param.b32    [func_retval0], 2;
; SM_90: st.param.b32    [func_retval0], 1;
; COMMON-NEXT: ret;
define i32 @qux() {
entry:
  %call = call i32 @__nvvm_reflect(ptr noundef @.str)
  switch i32 %call, label %sw.default [
    i32 900, label %sw.bb
    i32 700, label %sw.bb1
    i32 520, label %sw.bb2
  ]

sw.bb:
  br label %return

sw.bb1:
  br label %return

sw.bb2:
  br label %return

sw.default:
  br label %return

return:
  %retval = phi i32 [ 4, %sw.default ], [ 3, %sw.bb2 ], [ 2, %sw.bb1 ], [ 1, %sw.bb ]
  ret i32 %retval
}

; COMMON-LABEL: .visible .func  (.param .b32 func_retval0) phi()
; COMMON: st.param.b32    [func_retval0], 0;
; COMMON-NEXT: ret;
define float @phi() {
entry:
  %0 = call i32 @__nvvm_reflect(ptr @.str)
  %1 = icmp eq i32 %0, 0
  br i1 %1, label %if.then, label %if.else

if.then:
  br label %if.else

if.else:
  %.08 = phi float [ 0.000000e+00, %if.then ], [ 1.000000e+00, %entry ]
  %4 = fcmp ogt float %.08, 0.000000e+00
  br i1 %4, label %exit, label %if.exit

if.exit:
  br label %exit

exit:
  ret float 0.000000e+00
}

; COMMON-LABEL: .visible .func  (.param .b32 func_retval0) prop()
; SM_52: st.param.b32    [func_retval0], 3;
; SM_70: st.param.b32    [func_retval0], 2;
; SM_90: st.param.b32    [func_retval0], 1;
; COMMON-NEXT: ret;
define i32 @prop() {
entry:
  %call = call i32 @__nvvm_reflect(ptr @.str)
  %conv = zext i32 %call to i64
  %div = udiv i64 %conv, 100
  %cmp = icmp eq i64 %div, 9
  br i1 %cmp, label %if.then, label %if.else

if.then:
  br label %return

if.else:
  %div2 = udiv i64 %conv, 100
  %cmp3 = icmp eq i64 %div2, 7
  br i1 %cmp3, label %if.then5, label %if.else6

if.then5:
  br label %return

if.else6:
  %div7 = udiv i64 %conv, 100
  %cmp8 = icmp eq i64 %div7, 5
  br i1 %cmp8, label %if.then10, label %if.else11

if.then10:
  br label %return

if.else11:
  br label %return

return:
  %retval = phi i32 [ 1, %if.then ], [ 2, %if.then5 ], [ 3, %if.then10 ], [ 4, %if.else11 ]
  ret i32 %retval
}
