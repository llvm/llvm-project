; RUN: llc < %s -march=nvptx64 -mcpu=sm_52 -mattr=+ptx64 -O0 | FileCheck %s --check-prefix=SM_52
; RUN: llc < %s -march=nvptx64 -mcpu=sm_70 -mattr=+ptx64 -O0 | FileCheck %s --check-prefix=SM_70
; RUN: llc < %s -march=nvptx64 -mcpu=sm_90 -mattr=+ptx72 -O0 | FileCheck %s --check-prefix=SM_90

@.str = private unnamed_addr constant [12 x i8] c"__CUDA_ARCH\00"

declare i32 @__nvvm_reflect(ptr)

;      SM_52: .visible .func  (.param .b32 func_retval0) foo()
;      SM_52: mov.b32         %[[REG:.+]], 3;
; SM_52-NEXT: st.param.b32    [func_retval0+0], %[[REG:.+]];
; SM_52-NEXT: ret;
;
;      SM_70: .visible .func  (.param .b32 func_retval0) foo()
;      SM_70: mov.b32         %[[REG:.+]], 2;
; SM_70-NEXT: st.param.b32    [func_retval0+0], %[[REG:.+]];
; SM_70-NEXT: ret;
;
;      SM_90: .visible .func  (.param .b32 func_retval0) foo()
;      SM_90: mov.b32         %[[REG:.+]], 1;
; SM_90-NEXT: st.param.b32    [func_retval0+0], %[[REG:.+]];
; SM_90-NEXT: ret;
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

;      SM_52: .visible .func  (.param .b32 func_retval0) bar()
;      SM_52: mov.b32         %[[REG:.+]], 2;
; SM_52-NEXT: st.param.b32    [func_retval0+0], %[[REG:.+]];
; SM_52-NEXT: ret;
;
;      SM_70: .visible .func  (.param .b32 func_retval0) bar()
;      SM_70: mov.b32         %[[REG:.+]], 1;
; SM_70-NEXT: st.param.b32    [func_retval0+0], %[[REG:.+]];
; SM_70-NEXT: ret;
;
;      SM_90: .visible .func  (.param .b32 func_retval0) bar()
;      SM_90: mov.b32         %[[REG:.+]], 1;
; SM_90-NEXT: st.param.b32    [func_retval0+0], %[[REG:.+]];
; SM_90-NEXT: ret;
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
