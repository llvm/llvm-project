; RUN: opt -functionattrs -attributor -attributor-disable=false -S < %s | FileCheck %s
; RUN: opt -functionattrs -attributor -attributor-disable=false -attributor-verify=true -S < %s | FileCheck %s
;
; Test cases specifically designed for the "no-capture" argument attribute.
; We use FIXME's to indicate problems and missing attributes.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; TEST comparison against NULL
;
; int is_null_return(int *p) {
;   return p == 0;
; }
;
; FIXME: no-capture missing for %p
; CHECK: define i32 @is_null_return(i32* readnone %p)
define i32 @is_null_return(i32* %p) #0 {
entry:
  %cmp = icmp eq i32* %p, null
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; TEST comparison against NULL in control flow
;
; int is_null_control(int *p) {
;   if (p == 0)
;     return 1;
;   if (0 == p)
;     return 1;
;   return 0;
; }
;
; FIXME: no-capture missing for %p
; CHECK: define i32 @is_null_control(i32* readnone %p)
define i32 @is_null_control(i32* %p) #0 {
entry:
  %retval = alloca i32, align 4
  %cmp = icmp eq i32* %p, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 1, i32* %retval, align 4
  br label %return

if.end:                                           ; preds = %entry
  %cmp1 = icmp eq i32* null, %p
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:                                         ; preds = %if.end
  store i32 1, i32* %retval, align 4
  br label %return

if.end3:                                          ; preds = %if.end
  store i32 0, i32* %retval, align 4
  br label %return

return:                                           ; preds = %if.end3, %if.then2, %if.then
  %0 = load i32, i32* %retval, align 4
  ret i32 %0
}

; TEST singleton SCC
;
; double *srec0(double *a) {
;   srec0(a);
;   return 0;
; }
;
; CHECK: define noalias double* @srec0(double* nocapture readnone %a)
define double* @srec0(double* %a) #0 {
entry:
  %call = call double* @srec0(double* %a)
  ret double* null
}

; TEST singleton SCC with lots of nested recursive calls
;
; int* srec16(int* a) {
;   return srec16(srec16(srec16(srec16(
;          srec16(srec16(srec16(srec16(
;          srec16(srec16(srec16(srec16(
;          srec16(srec16(srec16(srec16(
;                        a
;          ))))))))))))))));
; }
;
; Other arguments are possible here due to the no-return behavior.
;
; FIXME: no-return missing
; CHECK: define noalias nonnull i32* @srec16(i32* nocapture readnone %a)
define i32* @srec16(i32* %a) #0 {
entry:
  %call = call i32* @srec16(i32* %a)
  %call1 = call i32* @srec16(i32* %call)
  %call2 = call i32* @srec16(i32* %call1)
  %call3 = call i32* @srec16(i32* %call2)
  %call4 = call i32* @srec16(i32* %call3)
  %call5 = call i32* @srec16(i32* %call4)
  %call6 = call i32* @srec16(i32* %call5)
  %call7 = call i32* @srec16(i32* %call6)
  %call8 = call i32* @srec16(i32* %call7)
  %call9 = call i32* @srec16(i32* %call8)
  %call10 = call i32* @srec16(i32* %call9)
  %call11 = call i32* @srec16(i32* %call10)
  %call12 = call i32* @srec16(i32* %call11)
  %call13 = call i32* @srec16(i32* %call12)
  %call14 = call i32* @srec16(i32* %call13)
  %call15 = call i32* @srec16(i32* %call14)
  ret i32* %call15
}

; TEST SCC with various calls, casts, and comparisons agains NULL
;
; FIXME: returned missing for %a
; FIXME: no-capture missing for %a
; CHECK: define float* @scc_A(i32* readnone %a)
;
; FIXME: returned missing for %a
; FIXME: no-capture missing for %a
; CHECK: define i64* @scc_B(double* readnone %a)
;
; FIXME: returned missing for %a
; FIXME: readnone missing for %s
; FIXME: no-capture missing for %a
; CHECK: define i8* @scc_C(i16* %a)
;
; float *scc_A(int *a) {
;   return (float*)(a ? (int*)scc_A((int*)scc_B((double*)scc_C((short*)a))) : a);
; }
;
; long *scc_B(double *a) {
;   return (long*)(a ? scc_C((short*)scc_B((double*)scc_A((int*)a))) : a);
; }
;
; void *scc_C(short *a) {
;   return scc_A((int*)(scc_C(a) ? scc_B((double*)a) : scc_C(a)));
; }
define float* @scc_A(i32* %a) {
entry:
  %tobool = icmp ne i32* %a, null
  br i1 %tobool, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  %0 = bitcast i32* %a to i16*
  %call = call i8* @scc_C(i16* %0)
  %1 = bitcast i8* %call to double*
  %call1 = call i64* @scc_B(double* %1)
  %2 = bitcast i64* %call1 to i32*
  %call2 = call float* @scc_A(i32* %2)
  %3 = bitcast float* %call2 to i32*
  br label %cond.end

cond.false:                                       ; preds = %entry
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32* [ %3, %cond.true ], [ %a, %cond.false ]
  %4 = bitcast i32* %cond to float*
  ret float* %4
}

define i64* @scc_B(double* %a) {
entry:
  %tobool = icmp ne double* %a, null
  br i1 %tobool, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  %0 = bitcast double* %a to i32*
  %call = call float* @scc_A(i32* %0)
  %1 = bitcast float* %call to double*
  %call1 = call i64* @scc_B(double* %1)
  %2 = bitcast i64* %call1 to i16*
  %call2 = call i8* @scc_C(i16* %2)
  br label %cond.end

cond.false:                                       ; preds = %entry
  %3 = bitcast double* %a to i8*
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i8* [ %call2, %cond.true ], [ %3, %cond.false ]
  %4 = bitcast i8* %cond to i64*
  ret i64* %4
}

define i8* @scc_C(i16* %a) {
entry:
  %call = call i8* @scc_C(i16* %a)
  %tobool = icmp ne i8* %call, null
  br i1 %tobool, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  %0 = bitcast i16* %a to double*
  %call1 = call i64* @scc_B(double* %0)
  %1 = bitcast i64* %call1 to i8*
  br label %cond.end

cond.false:                                       ; preds = %entry
  %call2 = call i8* @scc_C(i16* %a)
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i8* [ %1, %cond.true ], [ %call2, %cond.false ]
  %2 = bitcast i8* %cond to i32*
  %call3 = call float* @scc_A(i32* %2)
  %3 = bitcast float* %call3 to i8*
  ret i8* %3
}


; TEST call to external function, marked no-capture
;
; void external_no_capture(int /* no-capture */ *p);
; void test_external_no_capture(int *p) {
;   external_no_capture(p);
; }
;
; CHECK: define void @test_external_no_capture(i32* nocapture %p)
declare void @external_no_capture(i32* nocapture)

define void @test_external_no_capture(i32* %p) #0 {
entry:
  call void @external_no_capture(i32* %p)
  ret void
}

; TEST call to external var-args function, marked no-capture
;
; void test_var_arg_call(char *p, int a) {
;   printf(p, a);
; }
;
; CHECK: define void @test_var_arg_call(i8* nocapture %p, i32 %a)
define void @test_var_arg_call(i8* %p, i32 %a) #0 {
entry:
  %call = call i32 (i8*, ...) @printf(i8* %p, i32 %a)
  ret void
}

declare i32 @printf(i8* nocapture, ...)


; TEST "captured" only through return
;
; long *not_captured_but_returned_0(long *a) {
;   *a1 = 0;
;   return a;
; }
;
; There should *not* be a no-capture attribute on %a
; CHECK: define i64* @not_captured_but_returned_0(i64* returned %a)
define i64* @not_captured_but_returned_0(i64* %a) #0 {
entry:
  store i64 0, i64* %a, align 8
  ret i64* %a
}

; TEST "captured" only through return
;
; long *not_captured_but_returned_1(long *a) {
;   *(a+1) = 1;
;   return a + 1;
; }
;
; There should *not* be a no-capture attribute on %a
; CHECK: define nonnull i64* @not_captured_but_returned_1(i64* %a)
define i64* @not_captured_but_returned_1(i64* %a) #0 {
entry:
  %add.ptr = getelementptr inbounds i64, i64* %a, i64 1
  store i64 1, i64* %add.ptr, align 8
  ret i64* %add.ptr
}

; TEST calls to "captured" only through return functions
;
; void test_not_captured_but_returned_calls(long *a) {
;   not_captured_but_returned_0(a);
;   not_captured_but_returned_1(a);
; }
;
; FIXME: no-capture missing for %a
; CHECK: define void @test_not_captured_but_returned_calls(i64* %a)
define void @test_not_captured_but_returned_calls(i64* %a) #0 {
entry:
  %call = call i64* @not_captured_but_returned_0(i64* %a)
  %call1 = call i64* @not_captured_but_returned_1(i64* %a)
  ret void
}

; TEST "captured" only through transitive return
;
; long* negative_test_not_captured_but_returned_call_0a(long *a) {
;   return not_captured_but_returned_0(a);
; }
;
; There should *not* be a no-capture attribute on %a
; CHECK: define i64* @negative_test_not_captured_but_returned_call_0a(i64* returned %a)
define i64* @negative_test_not_captured_but_returned_call_0a(i64* %a) #0 {
entry:
  %call = call i64* @not_captured_but_returned_0(i64* %a)
  ret i64* %call
}

; TEST captured through write
;
; void negative_test_not_captured_but_returned_call_0b(long *a) {
;   *a = (long)not_captured_but_returned_0(a);
; }
;
; There should *not* be a no-capture attribute on %a
; CHECK: define void @negative_test_not_captured_but_returned_call_0b(i64* %a)
define void @negative_test_not_captured_but_returned_call_0b(i64* %a) #0 {
entry:
  %call = call i64* @not_captured_but_returned_0(i64* %a)
  %0 = ptrtoint i64* %call to i64
  store i64 %0, i64* %a, align 8
  ret void
}

; TEST "captured" only through transitive return
;
; long* negative_test_not_captured_but_returned_call_1a(long *a) {
;   return not_captured_but_returned_1(a);
; }
;
; There should *not* be a no-capture attribute on %a
; CHECK: define nonnull i64* @negative_test_not_captured_but_returned_call_1a(i64* %a)
define i64* @negative_test_not_captured_but_returned_call_1a(i64* %a) #0 {
entry:
  %call = call i64* @not_captured_but_returned_1(i64* %a)
  ret i64* %call
}

; TEST captured through write
;
; void negative_test_not_captured_but_returned_call_1b(long *a) {
;   *a = (long)not_captured_but_returned_1(a);
; }
;
; There should *not* be a no-capture attribute on %a
; CHECK: define void @negative_test_not_captured_but_returned_call_1b(i64* %a)
define void @negative_test_not_captured_but_returned_call_1b(i64* %a) #0 {
entry:
  %call = call i64* @not_captured_but_returned_1(i64* %a)
  %0 = ptrtoint i64* %call to i64
  store i64 %0, i64* %call, align 8
  ret void
}

; TEST return argument or unknown call result
;
; int* ret_arg_or_unknown(int* b) {
;   if (b == 0)
;     return b;
;   return unknown();
; }
;
; Verify we do *not* assume b is returned or not captured.
;
; CHECK:     define i32* @ret_arg_or_unknown(i32* readnone %b)
; CHECK:     define i32* @ret_arg_or_unknown_through_phi(i32* readnone %b)
declare i32* @unknown()

define i32* @ret_arg_or_unknown(i32* %b) #0 {
entry:
  %cmp = icmp eq i32* %b, null
  br i1 %cmp, label %ret_arg, label %ret_unknown

ret_arg:
  ret i32* %b

ret_unknown:
  %call = call i32* @unknown()
  ret i32* %call
}

define i32* @ret_arg_or_unknown_through_phi(i32* %b) #0 {
entry:
  %cmp = icmp eq i32* %b, null
  br i1 %cmp, label %ret_arg, label %ret_unknown

ret_arg:
  br label %r

ret_unknown:
  %call = call i32* @unknown()
  br label %r

r:
  %phi = phi i32* [ %b, %ret_arg ], [ %call, %ret_unknown ]
  ret i32* %phi
}


; TEST not captured by readonly external function
;
; CHECK: define void @not_captured_by_readonly_call(i32* nocapture %b)
declare i32* @readonly_unknown(i32*, i32*) readonly

define void @not_captured_by_readonly_call(i32* %b) #0 {
entry:
  %call = call i32* @readonly_unknown(i32* %b, i32* %b)
  ret void
}


; TEST not captured by readonly external function if return chain is known
;
; Make sure the returned flag on %r is strong enough to justify nocapture on %b but **not** on %r.
;
; FIXME: The "returned" information is not propagated to the fullest extend causing us to miss "nocapture" on %b in the following:
; CHECK: define i32* @not_captured_by_readonly_call_not_returned_either1(i32* readonly %b, i32* readonly returned %r)
;
; CHECK: define i32* @not_captured_by_readonly_call_not_returned_either2(i32* readonly %b, i32* readonly returned %r)
; CHECK: define i32* @not_captured_by_readonly_call_not_returned_either3(i32* readonly %b, i32* readonly returned %r)
;
; FIXME: The "nounwind" information is not derived to the fullest extend causing us to miss "nocapture" on %b in the following:
; CHECK: define i32* @not_captured_by_readonly_call_not_returned_either4(i32* readonly %b, i32* readonly returned %r)
define i32* @not_captured_by_readonly_call_not_returned_either1(i32* %b, i32* returned %r) #0 {
entry:
  %call = call i32* @readonly_unknown(i32* %b, i32* %r) nounwind
  ret i32* %call
}

declare i32* @readonly_unknown_r1a(i32*, i32* returned) readonly
define i32* @not_captured_by_readonly_call_not_returned_either2(i32* %b, i32* %r) #0 {
entry:
  %call = call i32* @readonly_unknown_r1a(i32* %b, i32* %r) nounwind
  ret i32* %call
}

declare i32* @readonly_unknown_r1b(i32*, i32* returned) readonly nounwind
define i32* @not_captured_by_readonly_call_not_returned_either3(i32* %b, i32* %r) #0 {
entry:
  %call = call i32* @readonly_unknown_r1b(i32* %b, i32* %r)
  ret i32* %call
}

define i32* @not_captured_by_readonly_call_not_returned_either4(i32* %b, i32* %r) #0 {
entry:
  %call = call i32* @readonly_unknown_r1a(i32* %b, i32* %r)
  ret i32* %call
}

attributes #0 = { noinline nounwind uwtable }
