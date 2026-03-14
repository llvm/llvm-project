; RUN: llc -verify-machineinstrs < %s | FileCheck %s
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.ref_s.1.49.91.115.121.139.145.151.157.163.169.175.181.211 = type { %union.v.0.48.90.114.120.138.144.150.156.162.168.174.180.210, i16, i16 }
%union.v.0.48.90.114.120.138.144.150.156.162.168.174.180.210 = type { i64 }
%struct.stream_s.5.53.95.119.125.143.149.155.161.167.173.179.185.215 = type { ptr, ptr, ptr, i32, i8, i8, i64, %struct.stream_procs.2.50.92.116.122.140.146.152.158.164.170.176.182.212, i32, ptr, ptr, i16, i32 }
%struct.stream_procs.2.50.92.116.122.140.146.152.158.164.170.176.182.212 = type { ptr, ptr, ptr, ptr, ptr, ptr }
%struct._IO_FILE.4.52.94.118.124.142.148.154.160.166.172.178.184.214 = type { i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, [1 x i8], ptr, i64, ptr, ptr, ptr, ptr, i64, i32, [20 x i8] }
%struct._IO_marker.3.51.93.117.123.141.147.153.159.165.171.177.183.213 = type { ptr, ptr, i32 }

@special_ops = external global [7 x ptr], align 8
@ostack = external global [520 x %struct.ref_s.1.49.91.115.121.139.145.151.157.163.169.175.181.211], align 8
@osbot = external global ptr, align 8
@osp = external global ptr, align 8
@ostop = external global ptr, align 8
@osp_nargs = external global [6 x ptr], align 8
@estack = external global [150 x %struct.ref_s.1.49.91.115.121.139.145.151.157.163.169.175.181.211], align 8
@esp = external global ptr, align 8
@estop = external global ptr, align 8
@dstack = external global [20 x %struct.ref_s.1.49.91.115.121.139.145.151.157.163.169.175.181.211], align 8
@dsp = external global ptr, align 8
@dstop = external global ptr, align 8
@name_errordict = external global %struct.ref_s.1.49.91.115.121.139.145.151.157.163.169.175.181.211
@name_ErrorNames = external global %struct.ref_s.1.49.91.115.121.139.145.151.157.163.169.175.181.211
@error_object = external global %struct.ref_s.1.49.91.115.121.139.145.151.157.163.169.175.181.211, align 8

declare i32 @zadd(ptr)

declare i32 @zdup(ptr)

declare i32 @zexch(ptr)

declare i32 @zifelse(ptr)

declare i32 @zle(ptr)

declare i32 @zpop(ptr)

declare i32 @zsub(ptr)

declare void @interp_init(i32) nounwind

declare void @interp_fix_op(ptr nocapture) nounwind

define i32 @interpret(ptr %pref, i32 %user_errors) nounwind {
entry:
  %erref = alloca %struct.ref_s.1.49.91.115.121.139.145.151.157.163.169.175.181.211, align 8
  br i1 undef, label %retry.us, label %retry

retry.us:                                         ; preds = %if.end18, %retry, %retry, %retry, %retry, %entry
  ret i32 undef

retry:                                            ; preds = %if.end18, %entry
  %0 = phi ptr [ null, %entry ], [ %erref, %if.end18 ]
  %call = call i32 @interp(ptr %0)
  switch i32 %call, label %if.end18 [
    i32 -3, label %retry.us
    i32 -5, label %retry.us
    i32 -16, label %retry.us
    i32 -25, label %retry.us
  ]

if.end18:                                         ; preds = %retry
  br i1 false, label %retry.us, label %retry
}

; CHECK: @interpret

declare i32 @interp_exit(ptr nocapture) nounwind readnone

declare i32 @interp(ptr) nounwind

declare i32 @dict_lookup(ptr, ptr, ptr, ptr)

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind

declare i32 @obj_compare(...)

declare i32 @file_check_read(...)

declare i32 @scan_token(...)

declare i32 @file_close(...)

declare void @sread_string(ptr, ptr, i32)
