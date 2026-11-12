;;  Note: The checks for this test are manually generated. Please do not
;;        run a script to update these checks.

; REQUIRES: asserts

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr8 \
; RUN:   --stats -ppc-min-jump-table-entries=4 < %s 2>&1 | FileCheck %s --check-prefix=AIX
; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr8 \
; RUN:   --stats -ppc-min-jump-table-entries=4 < %s 2>&1 | FileCheck %s --check-prefix=AIX
; RUN: llc -verify-machineinstrs -mtriple powerpc64-unknown-linux -mcpu=pwr8 \
; RUN:   --stats -ppc-min-jump-table-entries=4 < %s 2>&1 | FileCheck %s --check-prefix=LINUX
; RUN: llc -verify-machineinstrs -mtriple powerpc64le-unknown-linux -mcpu=pwr8 \
; RUN:   --stats -ppc-min-jump-table-entries=4 < %s 2>&1 | FileCheck %s --check-prefix=LINUX
; RUN: llc -verify-machineinstrs -mtriple powerpc64-unknown-linux -mcpu=pwr8 \
; RUN:   -code-model=large --stats -ppc-min-jump-table-entries=4 < %s 2>&1 | FileCheck %s --check-prefix=LINUXLARGE
; RUN: llc -verify-machineinstrs -mtriple powerpc64le-unknown-linux -mcpu=pwr8 \
; RUN:   -code-model=large --stats -ppc-min-jump-table-entries=4 < %s 2>&1 | FileCheck %s --check-prefix=LINUXLARGE


; The purpose of this test is to check that the statistics about the TOC are
; are collected correctly. This test tries to have all of the different types 
; of TOC entries.

; AIX: Statistics Collected
; AIX:   1 asmprinter           - Number of Block Address TOC Entries.
; AIX:  13 asmprinter           - Number of Constant Pool TOC Entries.
; AIX:   1 asmprinter           - Number of EH Block TOC Entries.
; AIX:  25 asmprinter           - Number of Total TOC Entries Emitted.
; AIX:   3 asmprinter           - Number of External Linkage Global TOC Entries.
; AIX:   2 asmprinter           - Number of Internal Linkage Global TOC Entries.
; AIX:   1 asmprinter           - Number of Jump Table TOC Entries.
; AIX:   4 asmprinter           - Number of Thread Local TOC Entries.

; LINUX: Statistics Collected
; LINUX:   1 asmprinter            - Number of Block Address TOC Entries.
; LINUX:   5 asmprinter            - Number of Total TOC Entries Emitted.
; LINUX:   3 asmprinter            - Number of External Linkage Global TOC Entries.
; LINUX:   1 asmprinter            - Number of Jump Table TOC Entries.

; LINUXLARGE: Statistics Collected
; LINUXLARGE:  1 asmprinter            - Number of Block Address TOC Entries.
; LINUXLARGE: 13 asmprinter            - Number of Constant Pool TOC Entries.
; LINUXLARGE: 20 asmprinter            - Number of Total TOC Entries Emitted.
; LINUXLARGE:  3 asmprinter            - Number of External Linkage Global TOC Entries.
; LINUXLARGE:  2 asmprinter            - Number of Internal Linkage Global TOC Entries.
; LINUXLARGE:  1 asmprinter            - Number of Jump Table TOC Entries.


@gDouble = local_unnamed_addr global double 0.000000e+00, align 8
@TLS1 = thread_local global i32 0, align 4
@TLS2 = external thread_local global float, align 4
@_ZTIi = external constant ptr
@_ZL2G4 = internal unnamed_addr global i32 0, align 4
@_ZZ9incrementvE7Element = internal unnamed_addr global i32 0, align 4

define noundef double @testd1() local_unnamed_addr {
entry:
  ret double 3.784320e+02
}

define noundef float @testf1() local_unnamed_addr {
entry:
  ret float 0x40039999A0000000
}

define noundef double @testd2() local_unnamed_addr {
entry:
  ret double 6.920000e+00
}

define noundef <4 x i32> @testv1() local_unnamed_addr {
entry:
  ret <4 x i32> <i32 13, i32 78, i32 -13, i32 100>
}

define noundef double @testd3() local_unnamed_addr {
entry:
  %call = tail call noundef double @calleeddf(double noundef 4.582600e+01, double noundef 0x40564F0A3D70A3D7, float noundef 0x402225E360000000)
  ret double %call
}

declare noundef double @calleeddf(double noundef, double noundef, float noundef) local_unnamed_addr

define noundef i64 @testi() local_unnamed_addr {
entry:
  ret i64 893471915
}

define noundef double @testld1() local_unnamed_addr {
entry:
  ret double 0x417179806D5CDEBE
}

define noundef signext i32 @testJT(i32 noundef signext %in) {
entry:
  switch i32 %in, label %sw.epilog [
    i32 45, label %return
    i32 86, label %sw.bb1
    i32 91, label %sw.bb2
    i32 101, label %sw.bb3
    i32 107, label %sw.bb4
    i32 76832712, label %sw.bb5
  ]

sw.bb1:                                           ; preds = %entry
  br label %return

sw.bb2:                                           ; preds = %entry
  br label %return

sw.bb3:                                           ; preds = %entry
  br label %return

sw.bb4:                                           ; preds = %entry
  br label %return

sw.bb5:                                           ; preds = %entry
  br label %return

sw.epilog:                                        ; preds = %entry
  br label %return

return:                                           ; preds = %entry, %sw.epilog, %sw.bb5, %sw.bb4, %sw.bb3, %sw.bb2, %sw.bb1
  %retval.0 = phi i32 [ 0, %sw.epilog ], [ 222, %sw.bb5 ], [ 15, %sw.bb4 ], [ 11, %sw.bb3 ], [ 17, %sw.bb2 ], [ 16, %sw.bb1 ], [ 19, %entry ]
  ret i32 %retval.0
}

define void @setGDouble(double noundef %a) local_unnamed_addr {
entry:
  store double %a, ptr @gDouble, align 8
  ret void
}

define noundef double @getGDouble() local_unnamed_addr {
entry:
  %0 = load double, ptr @gDouble, align 8
  ret double %0
}

define noundef signext i32 @testTLS(i32 noundef signext %a, i32 noundef signext %b, ptr nocapture noundef readonly %fIn) {
entry:
  %add = add nsw i32 %b, %a
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TLS1)
  store i32 %add, ptr %0, align 4
  tail call void @calleef(ptr noundef nonnull @testJT)
  %1 = load i32, ptr %0, align 4
  %call = tail call noundef signext i32 %fIn(i32 noundef signext %1)
  %2 = load i32, ptr %0, align 4
  %add1 = add nsw i32 %2, %call
  ret i32 %add1
}

define float @getTLS2() local_unnamed_addr {
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TLS2)
  %1 = load float, ptr %0, align 4
  ret float %1
}

declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull)
declare void @calleef(ptr noundef) local_unnamed_addr

define noundef signext i32 @testEH(i32 noundef signext %i) personality ptr @__xlcxx_personality_v1 {
entry:
  %cmp = icmp slt i32 %i, 1
  br i1 %cmp, label %if.then, label %try.cont

if.then:                                          ; preds = %entry
  %exception = tail call ptr @__cxa_allocate_exception(i64 4)
  store i32 10, ptr %exception, align 16
  invoke void @__cxa_throw(ptr nonnull %exception, ptr nonnull @_ZTIi, ptr null)
          to label %unreachable unwind label %lpad

lpad:                                             ; preds = %if.then
  %0 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIi
  %1 = extractvalue { ptr, i32 } %0, 1
  %2 = tail call i32 @llvm.eh.typeid.for(ptr nonnull @_ZTIi)
  %matches = icmp eq i32 %1, %2
  br i1 %matches, label %catch, label %ehcleanup

catch:                                            ; preds = %lpad
  %3 = extractvalue { ptr, i32 } %0, 0
  %4 = tail call ptr @__cxa_begin_catch(ptr %3)
  tail call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %entry, %catch
  %ret.0 = phi i32 [ 1, %catch ], [ 0, %entry ]
  ret i32 %ret.0

ehcleanup:                                        ; preds = %lpad
  resume { ptr, i32 } %0

unreachable:                                      ; preds = %if.then
  unreachable
}

declare ptr @__cxa_allocate_exception(i64) local_unnamed_addr
declare void @__cxa_throw(ptr, ptr, ptr) local_unnamed_addr
declare i32 @__xlcxx_personality_v1(...)
declare i32 @llvm.eh.typeid.for(ptr)
declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr
declare void @__cxa_end_catch() local_unnamed_addr

define noundef double @testISel1(i32 noundef signext %i) {
entry:
  %cmp = icmp slt i32 %i, 0
  %cmp1.not = icmp eq i32 %i, 0
  %. = select i1 %cmp1.not, double 1.618030e+00, double 2.718280e+00
  %retval.0 = select i1 %cmp, double 3.141590e+00, double %.
  ret double %retval.0
}

define noundef double @testISel2(i32 noundef signext %i) {
entry:
  %cmp = icmp slt i32 %i, 0
  br i1 %cmp, label %return, label %if.else

if.else:                                          ; preds = %entry
  %cmp1 = icmp ugt i32 %i, 30
  br i1 %cmp1, label %return, label %if.else3

if.else3:                                         ; preds = %if.else
  %cmp4 = icmp ugt i32 %i, 5
  %. = select i1 %cmp4, double 8.644600e+00, double 1.618030e+00
  br label %return

return:                                           ; preds = %if.else3, %if.else, %entry
  %retval.0 = phi double [ 3.141590e+00, %entry ], [ 2.718280e+00, %if.else ], [ %., %if.else3 ]
  ret double %retval.0
}


define ptr @testBlockAddr() {
entry:
  br label %here

here:
  ret ptr blockaddress(@testBlockAddr, %here)
}

define noundef signext i32 @_Z5getG4v() local_unnamed_addr {
entry:
  %0 = load i32, ptr @_ZL2G4, align 4
  ret i32 %0
}

define void @_Z5setG4i(i32 noundef signext %value) local_unnamed_addr {
entry:
  store i32 %value, ptr @_ZL2G4, align 4
  ret void
}

define noundef signext i32 @_Z9incrementv() local_unnamed_addr {
entry:
  %0 = load i32, ptr @_ZZ9incrementvE7Element, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr @_ZZ9incrementvE7Element, align 4
  ret i32 %inc
}
