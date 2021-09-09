; RUN: opt %loadPolly -polly-scops -polly-codegen -S < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.node = type { i32 (...)**, %class.node* }

define void @foobar(double* %A) {
if.end:
  br i1 undef, label %if.then29, label %lor.lhs.false

lor.lhs.false:
  %call25 = tail call i32 undef(%class.node* undef)
  br i1 undef, label %if.then29, label %if.end30

if.then29:
  br label %if.end30

if.end30:
  %tobool76.not = phi i1 [ false, %lor.lhs.false ], [ true, %if.then29 ]
  br label %if.end75

if.end75:
  br label %if.end79

if.end79:
  br label %if.then84

if.then84:
  br label %if.end91

if.end91:
  br i1 %tobool76.not, label %if.end98, label %if.then93

if.then93:
  store double 0.0, double* %A
  br label %if.end98

if.end98:
  %tobool131 = phi i1 [ false, %if.end91 ], [ true, %if.then93 ]
  ret void
}


; CHECK: polly.stmt.if.then93:
; CHECK:   store double 0.000000e+00, double* %A
; CHECK:   br label %polly.exiting

