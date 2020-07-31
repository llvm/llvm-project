; RUN: opt -function-attrs -S < %s | FileCheck %s --check-prefix=FNATTR
;
; Test cases specifically designed for the "returned" argument attribute.
; We use FIXME's to indicate problems and missing attributes.
;

; TEST SCC test returning an integer value argument
;
;
; FNATTR: define i32 @sink_r0(i32 returned %r)
; FNATTR: define i32 @scc_r1(i32 %a, i32 %r, i32 %b)
; FNATTR: define i32 @scc_r2(i32 %a, i32 %b, i32 %r)
; FNATTR: define i32 @scc_rX(i32 %a, i32 %b, i32 %r)
;
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
; FNATTR: define double* @ptr_sink_r0(double* readnone returned %r)
; FNATTR: define double* @ptr_scc_r1(double* %a, double* readnone %r, double* nocapture readnone %b)
; FNATTR: define double* @ptr_scc_r2(double* readnone %a, double* readnone %b, double* readnone %r)
;
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


; TEST a no-return singleton SCC
;
; int* rt0(int *a) {
;   return *a ? a : rt0(a);
; }
;
; FNATTR:  define i32* @rt0(i32* readonly %a)
define i32* @rt0(i32* %a) #0 {
entry:
  %v = load i32, i32* %a, align 4
  %tobool = icmp ne i32 %v, 0
  %call = call i32* @rt0(i32* %a)
  %sel = select i1 %tobool, i32* %a, i32* %call
  ret i32* %sel
}

; TEST a no-return singleton SCC
;
; int* rt1(int *a) {
;   return *a ? undef : rt1(a);
; }
;
; FNATTR:  define noalias i32* @rt1(i32* nocapture readonly %a)
define i32* @rt1(i32* %a) #0 {
entry:
  %v = load i32, i32* %a, align 4
  %tobool = icmp ne i32 %v, 0
  %call = call i32* @rt1(i32* %a)
  %sel = select i1 %tobool, i32* undef, i32* %call
  ret i32* %sel
}

; TEST another SCC test
;
; FNATTR:  define i32* @rt2_helper(i32* %a)
; FNATTR:  define i32* @rt2(i32* readnone %a, i32* readnone %b)
define i32* @rt2_helper(i32* %a) #0 {
entry:
  %call = call i32* @rt2(i32* %a, i32* %a)
  ret i32* %call
}

define i32* @rt2(i32* %a, i32 *%b) #0 {
entry:
  %cmp = icmp eq i32* %a, null
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %call = call i32* @rt2_helper(i32* %a)
  br label %if.end

if.end:
  %sel = phi i32* [ %b, %entry], [%call, %if.then]
  ret i32* %sel
}

; TEST another SCC test
;
; FNATTR:  define i32* @rt3_helper(i32* %a, i32* %b)
; FNATTR:  define i32* @rt3(i32* readnone %a, i32* readnone %b)
define i32* @rt3_helper(i32* %a, i32* %b) #0 {
entry:
  %call = call i32* @rt3(i32* %a, i32* %b)
  ret i32* %call
}

define i32* @rt3(i32* %a, i32 *%b) #0 {
entry:
  %cmp = icmp eq i32* %a, null
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %call = call i32* @rt3_helper(i32* %a, i32* %b)
  br label %if.end

if.end:
  %sel = phi i32* [ %b, %entry], [%call, %if.then]
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
;
; FNATTR:     define i32* @calls_unknown_fn(i32* readnone returned %r)
declare void @unknown_fn(i32* (i32*)*) #0

define i32* @calls_unknown_fn(i32* %r) #0 {
  tail call void @unknown_fn(i32* (i32*)* nonnull @calls_unknown_fn)
  ret i32* %r
}


; TEST return call to a function that might be redifined at link time
;
;  int *maybe_redefined_fn2(int *r) {
;    return r;
;  }
;
;  int *calls_maybe_redefined_fn2(int *r) {
;    return maybe_redefined_fn2(r);
;  }
;
; Verify the maybe-redefined function is not annotated:
;
;
; FNATTR:     define i32* @calls_maybe_redefined_fn2(i32* %r)
define linkonce_odr i32* @maybe_redefined_fn2(i32* %r) #0 {
entry:
  ret i32* %r
}

define i32* @calls_maybe_redefined_fn2(i32* %r) #0 {
entry:
  %call = call i32* @maybe_redefined_fn2(i32* %r)
  ret i32* %call
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
;
; FNATTR:     define double @select_and_phi(double %b)
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
;
; FNATTR:     define double @recursion_select_and_phi(i32 %a, double %b)
;
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
;
; FNATTR:     define double* @bitcast(i32* readnone %b)
;
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
;
; FNATTR:     define double* @bitcasts_select_and_phi(i32* readnone %b)
;
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
;
; FNATTR:     define double* @ret_arg_arg_undef(i32* readnone %b)
;
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
;
; FNATTR:     define double* @ret_undef_arg_arg(i32* readnone %b)
;
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
;
; FNATTR:     define double* @ret_undef_arg_undef(i32* readnone %b)
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
;   return unknown();
; }
;
; Verify we do not assume b is returned
;
; FNATTR:     define i32* @ret_arg_or_unknown(i32* %b)
; FNATTR:     define i32* @ret_arg_or_unknown_through_phi(i32* %b)
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

; TEST inconsistent IR in dead code.
;
; FNATTR:     define i32 @deadblockcall1(i32 %A)
; FNATTR:     define i32 @deadblockcall2(i32 %A)
; FNATTR:     define i32 @deadblockphi1(i32 %A)
; FNATTR:     define i32 @deadblockphi2(i32 %A)
define i32 @deadblockcall1(i32 %A) #0 {
entry:
  ret i32 %A
unreachableblock:
  %B = call i32 @deadblockcall1(i32 %B)
  ret i32 %B
}

declare i32 @deadblockcall_helper(i32 returned %A);

define i32 @deadblockcall2(i32 %A) #0 {
entry:
  ret i32 %A
unreachableblock1:
  %B = call i32 @deadblockcall_helper(i32 %B)
  ret i32 %B
unreachableblock2:
  %C = call i32 @deadblockcall1(i32 %C)
  ret i32 %C
}

define i32 @deadblockphi1(i32 %A) #0 {
entry:
  br label %r
unreachableblock1:
  %B = call i32 @deadblockcall_helper(i32 %B)
  ret i32 %B
unreachableblock2:
  %C = call i32 @deadblockcall1(i32 %C)
  br label %r
r:
  %PHI = phi i32 [%A, %entry], [%C, %unreachableblock2]
  ret i32 %PHI
}

define i32 @deadblockphi2(i32 %A) #0 {
entry:
  br label %r
unreachableblock1:
  %B = call i32 @deadblockcall_helper(i32 %B)
  br label %unreachableblock3
unreachableblock2:
  %C = call i32 @deadblockcall1(i32 %C)
  br label %unreachableblock3
unreachableblock3:
  %PHI1 = phi i32 [%B, %unreachableblock1], [%C, %unreachableblock2]
  br label %r
r:
  %PHI2 = phi i32 [%A, %entry], [%PHI1, %unreachableblock3]
  ret i32 %PHI2
}

declare void @noreturn() noreturn;

define i32 @deadblockphi3(i32 %A, i1 %c) #0 {
entry:
  br i1 %c, label %r, label %unreachablecall
unreachablecall:
  call void @noreturn();
  %B = call i32 @deadblockcall_helper(i32 0)
  br label %unreachableblock3
unreachableblock2:
  %C = call i32 @deadblockcall1(i32 %C)
  br label %unreachableblock3
unreachableblock3:
  %PHI1 = phi i32 [%B, %unreachablecall], [%C, %unreachableblock2]
  br label %r
r:
  %PHI2 = phi i32 [%A, %entry], [%PHI1, %unreachableblock3]
  ret i32 %PHI2
}

attributes #0 = { noinline nounwind uwtable }
