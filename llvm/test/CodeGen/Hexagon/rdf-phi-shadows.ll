; RUN: llc -march=hexagon -verify-machineinstrs < %s | FileCheck %s
; Check that we don't crash.
; CHECK: call printf
target triple = "hexagon"

%struct.1 = type { i16, i8, i32, i8*, i8*, i8*, i8*, i8*, i8*, i32* }
%struct.0 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32 }

declare void @foo(%struct.1*, %struct.0* readonly) local_unnamed_addr #0
declare zeroext i8 @bar() local_unnamed_addr #0
declare i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #0

@.str = private unnamed_addr constant [5 x i8] c"blah\00", align 1

define i32 @main(i32 %argc, i8** nocapture readonly %argv) local_unnamed_addr #0 {
entry:
  %t0 = alloca %struct.0, align 4
  br label %do.body

do.body:                                          ; preds = %if.end88.do.body_crit_edge, %entry
  %cond = icmp eq i32 undef, 0
  br i1 %cond, label %if.end49, label %if.then124

if.end49:                                         ; preds = %do.body
  br i1 undef, label %if.end55, label %if.then53

if.then53:                                        ; preds = %if.end49
  call void @foo(%struct.1* null, %struct.0* nonnull %t0)
  br label %if.end55

if.end55:                                         ; preds = %if.then53, %if.end49
  %call76 = call zeroext i8 @bar() #0
  switch i8 %call76, label %sw.epilog79 [
    i8 0, label %sw.bb77
    i8 3, label %sw.bb77
  ]

sw.bb77:                                          ; preds = %if.end55, %if.end55
  unreachable

sw.epilog79:                                      ; preds = %if.end55
  br i1 undef, label %if.end88, label %if.then81

if.then81:                                        ; preds = %sw.epilog79
  %div87 = fdiv float 0.000000e+00, undef
  br label %if.end88

if.end88:                                         ; preds = %if.then81, %sw.epilog79
  %t1 = phi float [ undef, %sw.epilog79 ], [ %div87, %if.then81 ]
  %div89 = fdiv float 1.000000e+00, %t1
  %.sroa.speculated = select i1 undef, float 0.000000e+00, float undef
  %conv108 = fpext float %.sroa.speculated to double
  %call113 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i32 0, i32 0), double undef, double %conv108, i64 undef, i32 undef) #0
  br i1 undef, label %if.end88.do.body_crit_edge, label %if.then124

if.end88.do.body_crit_edge:                       ; preds = %if.end88
  br label %do.body

if.then124:                                       ; preds = %if.end88, %do.body
  %t2 = phi float [ undef, %do.body ], [ %t1, %if.end88 ]
  ret i32 0
}

attributes #0 = { nounwind }
