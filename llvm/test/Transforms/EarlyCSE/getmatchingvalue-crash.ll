; RUN: opt -basic-aa -aa -memoryssa -early-cse-memssa -verify -S < %s | FileCheck %s

; Check that this doesn't crash. The crash only happens with expensive checks,
; but there doesn't seem to be a REQUIRES for that.

; CHECK: invoke void @f1

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%s.0 = type { %s.1 }
%s.1 = type { %s.2* }
%s.2 = type { %s.3, %s.6, %s.16 }
%s.3 = type { %s.4, %s.5 }
%s.4 = type { i32 (...)**, i64 }
%s.5 = type { i32 (...)** }
%s.6 = type <{ %s.7, %s.10, i8*, i32, [4 x i8] }>
%s.7 = type { i32 (...)**, %s.8, i8*, i8*, i8*, i8*, i8*, i8* }
%s.8 = type { %s.9* }
%s.9 = type opaque
%s.10 = type { %s.11 }
%s.11 = type { %s.12 }
%s.12 = type { %s.13 }
%s.13 = type { %s.14 }
%s.14 = type { %s.15 }
%s.15 = type { i64, i64, i8* }
%s.16 = type <{ %s.17, %s.18*, i32 }>
%s.17 = type { i32 (...)**, i32, i64, i64, i32, i32, i8*, i8*, void (i32, %s.17*, i32)**, i32*, i64, i64, i64*, i64, i64, i8**, i64, i64 }
%s.18 = type { i32 (...)**, %s.16 }
%s.19 = type { i8, %s.20 }
%s.20 = type { %s.21 }
%s.21 = type { %s.22*, %s.24, %s.26 }
%s.22 = type { %s.23* }
%s.23 = type <{ %s.22, %s.23*, %s.22*, i8, [7 x i8] }>
%s.24 = type { %s.25 }
%s.25 = type { %s.22 }
%s.26 = type { %s.27 }
%s.27 = type { i64 }

@g0 = external constant [1 x i8], align 1

declare i32 @f0(...)

; Function Attrs: uwtable
declare void @f1(%s.0* nocapture) align 2

declare void @f2(%s.10*, %s.2*)

declare void @f3(%s.10*, i8*, i32)

define i8* @f4(%s.19* %a0, i8* %a1, i32 %a2, i8* %a3) align 2 personality i8* bitcast (i32 (...)* @f0 to i8*) {
b0:
  %v0 = alloca %s.0, align 8
  br label %b1

b1:                                               ; preds = %b0
  invoke void @f5(%s.10* nonnull sret(%s.10) align 8 undef, i8* nonnull undef)
          to label %b6 unwind label %b3

b2:                                               ; preds = %b2
  %v1 = invoke nonnull align 8 dereferenceable(24) %s.10* @f6(%s.10* undef, i64 undef, i64 1)
          to label %b2 unwind label %b4

b3:                                               ; preds = %b1
  %v2 = landingpad { i8*, i32 }
          cleanup
  br label %b5

b4:                                               ; preds = %b2
  %v3 = landingpad { i8*, i32 }
          cleanup
  br label %b5

b5:                                               ; preds = %b4, %b3
  resume { i8*, i32 } undef

b6:                                               ; preds = %b1
  invoke void @f1(%s.0* nonnull %v0)
          to label %b8 unwind label %b7

b7:                                               ; preds = %b6
  %v4 = landingpad { i8*, i32 }
          cleanup
  br label %b20

b8:                                               ; preds = %b6
  invoke void @f2(%s.10* sret(%s.10) align 8 undef, %s.2* undef)
          to label %b10 unwind label %b14

b9:                                               ; No predecessors!
  br label %b16

b10:                                              ; preds = %b8
  %v6 = invoke i32 @f7(%s.10* nonnull undef, i64 0, i64 -1, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @g0, i64 0, i64 0), i64 undef)
          to label %b12 unwind label %b11

b11:                                              ; preds = %b10
  %v7 = landingpad { i8*, i32 }
          catch i8* null
  unreachable

b12:                                              ; preds = %b10
  invoke void @f3(%s.10* nonnull sret(%s.10) align 8 undef, i8* %a1, i32 %a2)
          to label %b13 unwind label %b15

b13:                                              ; preds = %b12
  unreachable

b14:                                              ; preds = %b8
  %v8 = landingpad { i8*, i32 }
          cleanup
  br label %b16

b15:                                              ; preds = %b12
  %v9 = landingpad { i8*, i32 }
          cleanup
  br label %b16

b16:                                              ; preds = %b15, %b14, %b9
  %v10 = getelementptr inbounds %s.0, %s.0* %v0, i64 0, i32 0
  %v11 = getelementptr inbounds %s.1, %s.1* %v10, i64 0, i32 0
  br label %b17

b17:                                              ; preds = %b16
  %v12 = load %s.2*, %s.2** %v11, align 8
  br label %b18

b18:                                              ; preds = %b17
  call void undef(%s.2* nonnull %v12)
  br label %b19

b19:                                              ; preds = %b18
  store %s.2* null, %s.2** %v11, align 8
  br label %b20

b20:                                              ; preds = %b19, %b7
  resume { i8*, i32 } undef
}

declare hidden void @f5(%s.10*, i8*)

declare %s.10* @f6(%s.10*, i64, i64)

declare i32 @f7(%s.10*, i64, i64, i8*, i64)
