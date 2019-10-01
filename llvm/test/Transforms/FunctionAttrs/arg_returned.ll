; RUN: opt -functionattrs -attributor -attributor-disable=false -S < %s | FileCheck %s
; RUN: opt -functionattrs -attributor -attributor-disable=false -attributor-verify=true -S < %s | FileCheck %s
;
; Test cases specifically designed for the "returned" argument attribute.
; We use FIXME's to indicate problems and missing attributes.
;

; TEST SCC test returning an integer value argument
;
; CHECK: Function Attrs: noinline norecurse nounwind readnone uwtable
; CHECK: define i32 @sink_r0(i32 returned %r)
;
; FIXME: returned on %r missing:
; CHECK: Function Attrs: noinline nounwind readnone uwtable
; CHECK: define i32 @scc_r1(i32 %a, i32 %r, i32 %b)
;
; FIXME: returned on %r missing:
; CHECK: Function Attrs: noinline nounwind readnone uwtable
; CHECK: define i32 @scc_r2(i32 %a, i32 %b, i32 %r)
;
; int scc_r1(int a, int b, int r);
; int scc_r2(int a, int b, int r);
;
; __attribute__((noinline)) int sink_r0(int r) {
;   return r;
; }
;
; __attribute__((noinline)) int scc_r1(int a, int r, int b) {
;   return scc_r2(r, a, sink_r0(r));
; }
;
; __attribute__((noinline)) int scc_r2(int a, int b, int r) {
;   if (a > b)
;     return scc_r2(b, a, sink_r0(r));
;   if (a < b)
;     return scc_r1(sink_r0(b), scc_r2(scc_r1(a, b, r), scc_r1(a, scc_r2(r, r, r), r), scc_r2(a, b, r)), scc_r1(a, b, r));
;   return a == b ? r : scc_r2(a, b, r);
; }
; __attribute__((noinline)) int scc_rX(int a, int b, int r) {
;   if (a > b)
;     return scc_r2(b, a, sink_r0(r));
;   if (a < b)                                                                         // V Diff to scc_r2
;     return scc_r1(sink_r0(b), scc_r2(scc_r1(a, b, r), scc_r1(a, scc_r2(r, r, r), r), scc_r1(a, b, r)), scc_r1(a, b, r));
;   return a == b ? r : scc_r2(a, b, r);
; }
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define i32 @sink_r0(i32 %r) #0 {
entry:
  ret i32 %r
}

define i32 @scc_r1(i32 %a, i32 %r, i32 %b) #0 {
entry:
  %call = call i32 @sink_r0(i32 %r)
  %call1 = call i32 @scc_r2(i32 %r, i32 %a, i32 %call)
  ret i32 %call1
}

define i32 @scc_r2(i32 %a, i32 %b, i32 %r) #0 {
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call = call i32 @sink_r0(i32 %r)
  %call1 = call i32 @scc_r2(i32 %b, i32 %a, i32 %call)
  br label %return

if.end:                                           ; preds = %entry
  %cmp2 = icmp slt i32 %a, %b
  br i1 %cmp2, label %if.then3, label %if.end12

if.then3:                                         ; preds = %if.end
  %call4 = call i32 @sink_r0(i32 %b)
  %call5 = call i32 @scc_r1(i32 %a, i32 %b, i32 %r)
  %call6 = call i32 @scc_r2(i32 %r, i32 %r, i32 %r)
  %call7 = call i32 @scc_r1(i32 %a, i32 %call6, i32 %r)
  %call8 = call i32 @scc_r2(i32 %a, i32 %b, i32 %r)
  %call9 = call i32 @scc_r2(i32 %call5, i32 %call7, i32 %call8)
  %call10 = call i32 @scc_r1(i32 %a, i32 %b, i32 %r)
  %call11 = call i32 @scc_r1(i32 %call4, i32 %call9, i32 %call10)
  br label %return

if.end12:                                         ; preds = %if.end
  %cmp13 = icmp eq i32 %a, %b
  br i1 %cmp13, label %cond.true, label %cond.false

cond.true:                                        ; preds = %if.end12
  br label %cond.end

cond.false:                                       ; preds = %if.end12
  %call14 = call i32 @scc_r2(i32 %a, i32 %b, i32 %r)
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %r, %cond.true ], [ %call14, %cond.false ]
  br label %return

return:                                           ; preds = %cond.end, %if.then3, %if.then
  %retval.0 = phi i32 [ %call1, %if.then ], [ %call11, %if.then3 ], [ %cond, %cond.end ]
  ret i32 %retval.0
}

define i32 @scc_rX(i32 %a, i32 %b, i32 %r) #0 {
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call = call i32 @sink_r0(i32 %r)
  %call1 = call i32 @scc_r2(i32 %b, i32 %a, i32 %call)
  br label %return

if.end:                                           ; preds = %entry
  %cmp2 = icmp slt i32 %a, %b
  br i1 %cmp2, label %if.then3, label %if.end12

if.then3:                                         ; preds = %if.end
  %call4 = call i32 @sink_r0(i32 %b)
  %call5 = call i32 @scc_r1(i32 %a, i32 %b, i32 %r)
  %call6 = call i32 @scc_r2(i32 %r, i32 %r, i32 %r)
  %call7 = call i32 @scc_r1(i32 %a, i32 %call6, i32 %r)
  %call8 = call i32 @scc_r1(i32 %a, i32 %b, i32 %r)
  %call9 = call i32 @scc_r2(i32 %call5, i32 %call7, i32 %call8)
  %call10 = call i32 @scc_r1(i32 %a, i32 %b, i32 %r)
  %call11 = call i32 @scc_r1(i32 %call4, i32 %call9, i32 %call10)
  br label %return

if.end12:                                         ; preds = %if.end
  %cmp13 = icmp eq i32 %a, %b
  br i1 %cmp13, label %cond.true, label %cond.false

cond.true:                                        ; preds = %if.end12
  br label %cond.end

cond.false:                                       ; preds = %if.end12
  %call14 = call i32 @scc_r2(i32 %a, i32 %b, i32 %r)
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %r, %cond.true ], [ %call14, %cond.false ]
  br label %return

return:                                           ; preds = %cond.end, %if.then3, %if.then
  %retval.0 = phi i32 [ %call1, %if.then ], [ %call11, %if.then3 ], [ %cond, %cond.end ]
  ret i32 %retval.0
}


; TEST SCC test returning a pointer value argument
;
; CHECK: Function Attrs: noinline norecurse nounwind readnone uwtable
; CHECK: define double* @ptr_sink_r0(double* readnone returned %r)
;
; FIXME: returned on %r missing:
; CHECK: Function Attrs: noinline nounwind readnone uwtable
; CHECK: define double* @ptr_scc_r1(double* %a, double* readnone %r, double* nocapture readnone %b)
;
; FIXME: returned on %r missing:
; CHECK: Function Attrs: noinline nounwind readnone uwtable
; CHECK: define double* @ptr_scc_r2(double* readnone %a, double* readnone %b, double* readnone %r)
;
; double* ptr_scc_r1(double* a, double* b, double* r);
; double* ptr_scc_r2(double* a, double* b, double* r);
;
; __attribute__((noinline)) double* ptr_sink_r0(double* r) {
;   return r;
; }
;
; __attribute__((noinline)) double* ptr_scc_r1(double* a, double* r, double* b) {
;   return ptr_scc_r2(r, a, ptr_sink_r0(r));
; }
;
; __attribute__((noinline)) double* ptr_scc_r2(double* a, double* b, double* r) {
;   if (a > b)
;     return ptr_scc_r2(b, a, ptr_sink_r0(r));
;   if (a < b)
;     return ptr_scc_r1(ptr_sink_r0(b), ptr_scc_r2(ptr_scc_r1(a, b, r), ptr_scc_r1(a, ptr_scc_r2(r, r, r), r), ptr_scc_r2(a, b, r)), ptr_scc_r1(a, b, r));
;   return a == b ? r : ptr_scc_r2(a, b, r);
; }
define double* @ptr_sink_r0(double* %r) #0 {
entry:
  ret double* %r
}

define double* @ptr_scc_r1(double* %a, double* %r, double* %b) #0 {
entry:
  %call = call double* @ptr_sink_r0(double* %r)
  %call1 = call double* @ptr_scc_r2(double* %r, double* %a, double* %call)
  ret double* %call1
}

define double* @ptr_scc_r2(double* %a, double* %b, double* %r) #0 {
entry:
  %cmp = icmp ugt double* %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call = call double* @ptr_sink_r0(double* %r)
  %call1 = call double* @ptr_scc_r2(double* %b, double* %a, double* %call)
  br label %return

if.end:                                           ; preds = %entry
  %cmp2 = icmp ult double* %a, %b
  br i1 %cmp2, label %if.then3, label %if.end12

if.then3:                                         ; preds = %if.end
  %call4 = call double* @ptr_sink_r0(double* %b)
  %call5 = call double* @ptr_scc_r1(double* %a, double* %b, double* %r)
  %call6 = call double* @ptr_scc_r2(double* %r, double* %r, double* %r)
  %call7 = call double* @ptr_scc_r1(double* %a, double* %call6, double* %r)
  %call8 = call double* @ptr_scc_r2(double* %a, double* %b, double* %r)
  %call9 = call double* @ptr_scc_r2(double* %call5, double* %call7, double* %call8)
  %call10 = call double* @ptr_scc_r1(double* %a, double* %b, double* %r)
  %call11 = call double* @ptr_scc_r1(double* %call4, double* %call9, double* %call10)
  br label %return

if.end12:                                         ; preds = %if.end
  %cmp13 = icmp eq double* %a, %b
  br i1 %cmp13, label %cond.true, label %cond.false

cond.true:                                        ; preds = %if.end12
  br label %cond.end

cond.false:                                       ; preds = %if.end12
  %call14 = call double* @ptr_scc_r2(double* %a, double* %b, double* %r)
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi double* [ %r, %cond.true ], [ %call14, %cond.false ]
  br label %return

return:                                           ; preds = %cond.end, %if.then3, %if.then
  %retval.0 = phi double* [ %call1, %if.then ], [ %call11, %if.then3 ], [ %cond, %cond.end ]
  ret double* %retval.0
}


; TEST a singleton SCC with a lot of recursive calls
;
; int* ret0(int *a) {
;   return *a ? a : ret0(ret0(ret0(...ret0(a)...)));
; }
;
; FIXME: returned on %a missing:
; CHECK: Function Attrs: noinline nounwind readonly uwtable
; CHECK: define i32* @ret0(i32* readonly %a)
define i32* @ret0(i32* %a) #0 {
entry:
  %v = load i32, i32* %a, align 4
  %tobool = icmp ne i32 %v, 0
  %call = call i32* @ret0(i32* %a)
  %call1 = call i32* @ret0(i32* %call)
  %call2 = call i32* @ret0(i32* %call1)
  %call3 = call i32* @ret0(i32* %call2)
  %call4 = call i32* @ret0(i32* %call3)
  %call5 = call i32* @ret0(i32* %call4)
  %call6 = call i32* @ret0(i32* %call5)
  %call7 = call i32* @ret0(i32* %call6)
  %call8 = call i32* @ret0(i32* %call7)
  %call9 = call i32* @ret0(i32* %call8)
  %call10 = call i32* @ret0(i32* %call9)
  %call11 = call i32* @ret0(i32* %call10)
  %call12 = call i32* @ret0(i32* %call11)
  %call13 = call i32* @ret0(i32* %call12)
  %call14 = call i32* @ret0(i32* %call13)
  %call15 = call i32* @ret0(i32* %call14)
  %call16 = call i32* @ret0(i32* %call15)
  %call17 = call i32* @ret0(i32* %call16)
  %sel = select i1 %tobool, i32* %a, i32* %call17
  ret i32* %sel
}


; TEST address taken function with call to an external functions
;
;  void unknown_fn(void *);
;
;  int* calls_unknown_fn(int *r) {
;    unknown_fn(&calls_unknown_fn);
;    return r;
;  }
;
; CHECK: Function Attrs: noinline nounwind uwtable
; CHECK: declare void @unknown_fn(i32* (i32*)*)
;
; CHECK: Function Attrs: noinline nounwind uwtable
; CHECK: define i32* @calls_unknown_fn(i32* readnone returned %r)
declare void @unknown_fn(i32* (i32*)*) #0

define i32* @calls_unknown_fn(i32* %r) #0 {
  tail call void @unknown_fn(i32* (i32*)* nonnull @calls_unknown_fn)
  ret i32* %r
}


; TEST call to a function that might be redifined at link time
;
;  int *maybe_redefined_fn(int *r) {
;    return r;
;  }
;
;  int *calls_maybe_redefined_fn(int *r) {
;    maybe_redefined_fn(r);
;    return r;
;  }
;
; Verify the maybe-redefined function is not annotated:
;
; CHECK: Function Attrs: noinline nounwind uwtable
; CHECK: define linkonce_odr i32* @maybe_redefined_fn(i32* %r)
;
; CHECK: Function Attrs: noinline nounwind uwtable
; CHECK: define i32* @calls_maybe_redefined_fn(i32* returned %r)
define linkonce_odr i32* @maybe_redefined_fn(i32* %r) #0 {
entry:
  ret i32* %r
}

define i32* @calls_maybe_redefined_fn(i32* %r) #0 {
entry:
  %call = call i32* @maybe_redefined_fn(i32* %r)
  ret i32* %r
}


; TEST returned argument goes through select and phi
;
; double select_and_phi(double b) {
;   double x = b;
;   if (b > 0)
;     x = b;
;   return b == 0? b : x;
; }
;
; FIXME: returned on %b missing:
; CHECK: Function Attrs: noinline norecurse nounwind readnone uwtable
; CHECK: define double @select_and_phi(double %b)
define double @select_and_phi(double %b) #0 {
entry:
  %cmp = fcmp ogt double %b, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %phi = phi double [ %b, %if.then ], [ %b, %entry ]
  %cmp1 = fcmp oeq double %b, 0.000000e+00
  %sel = select i1 %cmp1, double %b, double %phi
  ret double %sel
}


; TEST returned argument goes through recursion, select, and phi
;
; double recursion_select_and_phi(int a, double b) {
;   double x = b;
;   if (a-- > 0)
;     x = recursion_select_and_phi(a, b);
;   return b == 0? b : x;
; }
;
; FIXME: returned on %b missing:
; CHECK: Function Attrs: noinline nounwind readnone uwtable
; CHECK: define double @recursion_select_and_phi(i32 %a, double %b)
define double @recursion_select_and_phi(i32 %a, double %b) #0 {
entry:
  %dec = add nsw i32 %a, -1
  %cmp = icmp sgt i32 %a, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call = call double @recursion_select_and_phi(i32 %dec, double %b)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %phi = phi double [ %call, %if.then ], [ %b, %entry ]
  %cmp1 = fcmp oeq double %b, 0.000000e+00
  %sel = select i1 %cmp1, double %b, double %phi
  ret double %sel
}


; TEST returned argument goes through bitcasts
;
; double* bitcast(int* b) {
;   return (double*)b;
; }
;
; FIXME: returned on %b missing:
; CHECK: Function Attrs: noinline norecurse nounwind readnone uwtable
; CHECK: define double* @bitcast(i32* readnone %b)
define double* @bitcast(i32* %b) #0 {
entry:
  %bc0 = bitcast i32* %b to double*
  ret double* %bc0
}


; TEST returned argument goes through select and phi interleaved with bitcasts
;
; double* bitcasts_select_and_phi(int* b) {
;   double* x = b;
;   if (b == 0)
;     x = b;
;   return b != 0 ? b : x;
; }
;
; FIXME: returned on %b missing:
; CHECK: Function Attrs: noinline norecurse nounwind readnone uwtable
; CHECK: define double* @bitcasts_select_and_phi(i32* readnone %b)
define double* @bitcasts_select_and_phi(i32* %b) #0 {
entry:
  %bc0 = bitcast i32* %b to double*
  %cmp = icmp eq double* %bc0, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %bc1 = bitcast i32* %b to double*
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %phi = phi double* [ %bc1, %if.then ], [ %bc0, %entry ]
  %bc2 = bitcast double* %phi to i8*
  %bc3 = bitcast i32* %b to i8*
  %cmp2 = icmp ne double* %bc0, null
  %sel = select i1 %cmp2, i8* %bc2, i8* %bc3
  %bc4 = bitcast i8* %sel to double*
  ret double* %bc4
}


; TEST return argument or argument or undef
;
; double* ret_arg_arg_undef(int* b) {
;   if (b == 0)
;     return (double*)b;
;   if (b == 0)
;     return (double*)b;
;   /* return undef */
; }
;
; CHECK: Function Attrs: noinline norecurse nounwind readnone uwtable
; CHECK:     define double* @ret_arg_arg_undef(i32* readnone %b)
define double* @ret_arg_arg_undef(i32* %b) #0 {
entry:
  %bc0 = bitcast i32* %b to double*
  %cmp = icmp eq double* %bc0, null
  br i1 %cmp, label %ret_arg0, label %if.end

ret_arg0:
  %bc1 = bitcast i32* %b to double*
  ret double* %bc1

if.end:
  br i1 %cmp, label %ret_arg1, label %ret_undef

ret_arg1:
  ret double* %bc0

ret_undef:
  ret double *undef
}


; TEST return undef or argument or argument
;
; double* ret_undef_arg_arg(int* b) {
;   if (b == 0)
;     return (double*)b;
;   if (b == 0)
;     return (double*)b;
;   /* return undef */
; }
;
; CHECK: Function Attrs: noinline norecurse nounwind readnone uwtable
; CHECK:     define double* @ret_undef_arg_arg(i32* readnone %b)
define double* @ret_undef_arg_arg(i32* %b) #0 {
entry:
  %bc0 = bitcast i32* %b to double*
  %cmp = icmp eq double* %bc0, null
  br i1 %cmp, label %ret_undef, label %if.end

ret_undef:
  ret double *undef

if.end:
  br i1 %cmp, label %ret_arg0, label %ret_arg1

ret_arg0:
  ret double* %bc0

ret_arg1:
  %bc1 = bitcast i32* %b to double*
  ret double* %bc1
}


; TEST return undef or argument or undef
;
; double* ret_undef_arg_undef(int* b) {
;   if (b == 0)
;     /* return undef */
;   if (b == 0)
;     return (double*)b;
;   /* return undef */
; }
;
; CHECK: Function Attrs: noinline norecurse nounwind readnone uwtable
; CHECK:     define double* @ret_undef_arg_undef(i32* readnone %b)
define double* @ret_undef_arg_undef(i32* %b) #0 {
entry:
  %bc0 = bitcast i32* %b to double*
  %cmp = icmp eq double* %bc0, null
  br i1 %cmp, label %ret_undef0, label %if.end

ret_undef0:
  ret double *undef

if.end:
  br i1 %cmp, label %ret_arg, label %ret_undef1

ret_arg:
  ret double* %bc0

ret_undef1:
  ret double *undef
}

; TEST return argument or unknown call result
;
; int* ret_arg_or_unknown(int* b) {
;   if (b == 0)
;     return b;
;   return unknown(b);
; }
;
; Verify we do not assume b is returned>
;
; CHECK:     define i32* @ret_arg_or_unknown(i32* %b)
; CHECK:     define i32* @ret_arg_or_unknown_through_phi(i32* %b)
declare i32* @unknown(i32*)

define i32* @ret_arg_or_unknown(i32* %b) #0 {
entry:
  %cmp = icmp eq i32* %b, null
  br i1 %cmp, label %ret_arg, label %ret_unknown

ret_arg:
  ret i32* %b

ret_unknown:
  %call = call i32* @unknown(i32* %b)
  ret i32* %call
}

define i32* @ret_arg_or_unknown_through_phi(i32* %b) #0 {
entry:
  %cmp = icmp eq i32* %b, null
  br i1 %cmp, label %ret_arg, label %ret_unknown

ret_arg:
  br label %r

ret_unknown:
  %call = call i32* @unknown(i32* %b)
  br label %r

r:
  %phi = phi i32* [ %b, %ret_arg ], [ %call, %ret_unknown ]
  ret i32* %phi
}

attributes #0 = { noinline nounwind uwtable }

; CHECK-NOT: attributes #
; CHECK-DAG: attributes #{{[0-9]*}} = { noinline norecurse nounwind readnone uwtable }
; CHECK-DAG: attributes #{{[0-9]*}} = { noinline nounwind readnone uwtable }
; CHECK-DAG: attributes #{{[0-9]*}} = { noinline nounwind readonly uwtable }
; CHECK-DAG: attributes #{{[0-9]*}} = { noinline nounwind uwtable }
; CHECK-NOT: attributes #
