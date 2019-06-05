; RUN: opt -functionattrs -enable-nonnull-arg-prop -attributor -attributor-disable=false -S < %s | FileCheck %s
; RUN: opt -functionattrs -enable-nonnull-arg-prop -attributor -attributor-disable=false -attributor-verify=true -S < %s | FileCheck %s
;
; This is an evolved example to stress test SCC parameter attribute propagation.
; The SCC in this test is made up of the following six function, three of which
; are internal and three externally visible:
;
; static int *internal_ret0_nw(int *n0, int *w0);
; static int *internal_ret1_rw(int *r0, int *w0);
; static int *internal_ret1_rrw(int *r0, int *r1, int *w0);
;        int *external_ret2_nrw(int *n0, int *r0, int *w0);
;        int *external_sink_ret2_nrw(int *n0, int *r0, int *w0);
;        int *external_source_ret2_nrw(int *n0, int *r0, int *w0);
;
; The top four functions call each other while the "sink" function will not
; call anything and the "source" function will not be called in this module.
; The names of the functions define the returned parameter (X for "_retX_"),
; as well as how the parameters are (transitively) used (n = readnone,
; r = readonly, w = writeonly).
;
; What we should see is something along the lines of:
;   1 - Number of functions marked as norecurse
;   6 - Number of functions marked argmemonly
;   6 - Number of functions marked as nounwind
;  16 - Number of arguments marked nocapture
;   4 - Number of arguments marked readnone
;   6 - Number of arguments marked writeonly
;   6 - Number of arguments marked readonly
;   6 - Number of arguments marked returned
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK: Function Attrs: nounwind
; CHECK-NEXT: define i32* @external_ret2_nrw(i32* %n0, i32* %r0, i32* %w0)
define i32* @external_ret2_nrw(i32* %n0, i32* %r0, i32* %w0) {
entry:
  %call = call i32* @internal_ret0_nw(i32* %n0, i32* %w0)
  %call1 = call i32* @internal_ret1_rrw(i32* %r0, i32* %r0, i32* %w0)
  %call2 = call i32* @external_sink_ret2_nrw(i32* %n0, i32* %r0, i32* %w0)
  %call3 = call i32* @internal_ret1_rw(i32* %r0, i32* %w0)
  ret i32* %call3
}

; CHECK: Function Attrs: nounwind
; CHECK-NEXT: define internal i32* @internal_ret0_nw(i32* %n0, i32* %w0)
define internal i32* @internal_ret0_nw(i32* %n0, i32* %w0) {
entry:
  %r0 = alloca i32, align 4
  %r1 = alloca i32, align 4
  %tobool = icmp ne i32* %n0, null
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  br label %return

if.end:                                           ; preds = %entry
  store i32 3, i32* %r0, align 4
  store i32 5, i32* %r1, align 4
  store i32 1, i32* %w0, align 4
  %call = call i32* @internal_ret1_rrw(i32* %r0, i32* %r1, i32* %w0)
  %call1 = call i32* @external_ret2_nrw(i32* %n0, i32* %r0, i32* %w0)
  %call2 = call i32* @external_ret2_nrw(i32* %n0, i32* %r1, i32* %w0)
  %call3 = call i32* @external_sink_ret2_nrw(i32* %n0, i32* %r0, i32* %w0)
  %call4 = call i32* @external_sink_ret2_nrw(i32* %n0, i32* %r1, i32* %w0)
  %call5 = call i32* @internal_ret0_nw(i32* %n0, i32* %w0)
  br label %return

return:                                           ; preds = %if.end, %if.then
  %retval.0 = phi i32* [ %call5, %if.end ], [ %n0, %if.then ]
  ret i32* %retval.0
}

; CHECK: Function Attrs: nounwind
; CHECK-NEXT: define internal i32* @internal_ret1_rrw(i32* %r0, i32* %r1, i32* %w0)
define internal i32* @internal_ret1_rrw(i32* %r0, i32* %r1, i32* %w0) {
entry:
  %0 = load i32, i32* %r0, align 4
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  br label %return

if.end:                                           ; preds = %entry
  %call = call i32* @internal_ret1_rw(i32* %r0, i32* %w0)
  %1 = load i32, i32* %r0, align 4
  %2 = load i32, i32* %r1, align 4
  %add = add nsw i32 %1, %2
  store i32 %add, i32* %w0, align 4
  %call1 = call i32* @internal_ret1_rw(i32* %r1, i32* %w0)
  %call2 = call i32* @internal_ret0_nw(i32* %r0, i32* %w0)
  %call3 = call i32* @internal_ret0_nw(i32* %w0, i32* %w0)
  %call4 = call i32* @external_ret2_nrw(i32* %r0, i32* %r1, i32* %w0)
  %call5 = call i32* @external_ret2_nrw(i32* %r1, i32* %r0, i32* %w0)
  %call6 = call i32* @external_sink_ret2_nrw(i32* %r0, i32* %r1, i32* %w0)
  %call7 = call i32* @external_sink_ret2_nrw(i32* %r1, i32* %r0, i32* %w0)
  %call8 = call i32* @internal_ret0_nw(i32* %r1, i32* %w0)
  br label %return

return:                                           ; preds = %if.end, %if.then
  %retval.0 = phi i32* [ %call8, %if.end ], [ %r1, %if.then ]
  ret i32* %retval.0
}

; CHECK: Function Attrs: norecurse nounwind
; CHECK-NEXT: define i32* @external_sink_ret2_nrw(i32* readnone %n0, i32* nocapture readonly %r0, i32* returned %w0)
define i32* @external_sink_ret2_nrw(i32* %n0, i32* %r0, i32* %w0) {
entry:
  %tobool = icmp ne i32* %n0, null
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  br label %return

if.end:                                           ; preds = %entry
  %0 = load i32, i32* %r0, align 4
  store i32 %0, i32* %w0, align 4
  br label %return

return:                                           ; preds = %if.end, %if.then
  ret i32* %w0
}

; CHECK: Function Attrs: nounwind
; CHECK-NEXT: define internal i32* @internal_ret1_rw(i32* %r0, i32* %w0)
define internal i32* @internal_ret1_rw(i32* %r0, i32* %w0) {
entry:
  %0 = load i32, i32* %r0, align 4
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  br label %return

if.end:                                           ; preds = %entry
  %call = call i32* @internal_ret1_rrw(i32* %r0, i32* %r0, i32* %w0)
  %1 = load i32, i32* %r0, align 4
  store i32 %1, i32* %w0, align 4
  %call1 = call i32* @internal_ret0_nw(i32* %r0, i32* %w0)
  %call2 = call i32* @internal_ret0_nw(i32* %w0, i32* %w0)
  %call3 = call i32* @external_sink_ret2_nrw(i32* %r0, i32* %r0, i32* %w0)
  %call4 = call i32* @external_ret2_nrw(i32* %r0, i32* %r0, i32* %w0)
  br label %return

return:                                           ; preds = %if.end, %if.then
  %retval.0 = phi i32* [ %call4, %if.end ], [ %w0, %if.then ]
  ret i32* %retval.0
}

; CHECK: Function Attrs: nounwind
; CHECK-NEXT: define i32* @external_source_ret2_nrw(i32* %n0, i32* %r0, i32* %w0)
define i32* @external_source_ret2_nrw(i32* %n0, i32* %r0, i32* %w0) {
entry:
  %call = call i32* @external_sink_ret2_nrw(i32* %n0, i32* %r0, i32* %w0)
  %call1 = call i32* @external_ret2_nrw(i32* %n0, i32* %r0, i32* %w0)
  ret i32* %call1
}

; Verify that we see only expected attribute sets, the above lines only check
; for a subset relation.
;
; CHECK-NOT: attributes #
; CHECK: attributes #{{.*}} = { nounwind }
; CHECK: attributes #{{.*}} = { norecurse nounwind }
; CHECK-NOT: attributes #
