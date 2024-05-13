; RUN: llc -march=hexagon -verify-machineinstrs -hexagon-initial-cfg-cleanup=0 < %s | FileCheck %s
; Verify that the code compiles successfully.
; CHECK: call printf

target triple = "hexagon"

%struct.0 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32 }

@.str.13 = external unnamed_addr constant [60 x i8], align 1

declare void @printf(ptr nocapture readonly, ...) local_unnamed_addr #0

declare void @danny() local_unnamed_addr #0
declare zeroext i8 @sammy() local_unnamed_addr #0

; Function Attrs: nounwind
define void @main() local_unnamed_addr #0 {
entry:
  br i1 undef, label %if.then8, label %if.end10

if.then8:                                         ; preds = %entry
  ret void

if.end10:                                         ; preds = %entry
  br label %do.body

do.body:                                          ; preds = %if.end88.do.body_crit_edge, %if.end10
  %cond = icmp eq i32 undef, 0
  br i1 %cond, label %if.end49, label %if.then124

if.end49:                                         ; preds = %do.body
  br i1 undef, label %if.end55, label %if.then53

if.then53:                                        ; preds = %if.end49
  call void @danny()
  br label %if.end55

if.end55:                                         ; preds = %if.then53, %if.end49
  %call76 = call zeroext i8 @sammy() #0
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
  %t.1 = phi float [ undef, %sw.epilog79 ], [ %div87, %if.then81 ]
  %div89 = fdiv float 1.000000e+00, %t.1
  %mul92 = fmul float undef, %div89
  %div93 = fdiv float %mul92, 1.000000e+06
  %conv107 = fpext float %div93 to double
  call void (ptr, ...) @printf(ptr @.str.13, double %conv107, double undef, i64 undef, i32 undef) #0
  br i1 undef, label %if.end88.do.body_crit_edge, label %if.then124

if.end88.do.body_crit_edge:                       ; preds = %if.end88
  br label %do.body

if.then124:                                       ; preds = %if.end88, %do.body
  unreachable
}


attributes #0 = { nounwind }

