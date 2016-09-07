; RUN: llc -march=hexagon -verify-machineinstrs < %s | FileCheck %s
; Check that we don't crash.
; CHECK: call foo

target triple = "hexagon"

%struct.1 = type { i16, i8, i32, i8*, i8*, i8*, i8*, i8*, i8*, i32* }
%struct.0 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32 }

declare void @foo(i8*, %struct.0*) local_unnamed_addr #0
declare void @bar(%struct.1*, %struct.0* readonly) local_unnamed_addr #0

define i32 @fred(i32 %argc, i8** nocapture readonly %argv) local_unnamed_addr #0 {
entry:
  br label %do.body

do.body:                                          ; preds = %if.end88.do.body_crit_edge, %entry
  %cond = icmp eq i32 undef, 0
  br i1 %cond, label %if.end49, label %if.then124

if.end49:                                         ; preds = %do.body
  call void @foo(i8* nonnull undef, %struct.0* nonnull undef) #0
  br i1 undef, label %if.end55, label %if.then53

if.then53:                                        ; preds = %if.end49
  call void @bar(%struct.1* null, %struct.0* nonnull undef)
  br label %if.end55

if.end55:                                         ; preds = %if.then53, %if.end49
  switch i8 undef, label %sw.epilog79 [
    i8 0, label %sw.bb77
    i8 3, label %sw.bb77
  ]

sw.bb77:                                          ; preds = %if.end55, %if.end55
  br label %sw.epilog79

sw.epilog79:                                      ; preds = %sw.bb77, %if.end55
  br i1 undef, label %if.end88, label %if.then81

if.then81:                                        ; preds = %sw.epilog79
  br label %if.end88

if.end88:                                         ; preds = %if.then81, %sw.epilog79
  store float 0.000000e+00, float* undef, align 4
  br i1 undef, label %if.end88.do.body_crit_edge, label %if.then124

if.end88.do.body_crit_edge:                       ; preds = %if.end88
  br label %do.body

if.then124:                                       ; preds = %if.end88, %do.body
  unreachable
}

attributes #0 = { nounwind }
