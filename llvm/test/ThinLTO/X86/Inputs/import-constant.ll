target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i32, i32, ptr }
%struct.Q = type { ptr }

@val = dso_local global i32 42, align 4
@_ZL3Obj = internal constant %struct.S { i32 4, i32 8, ptr @val }, align 8
@outer = dso_local local_unnamed_addr global %struct.Q { ptr @_ZL3Obj }, align 8

define dso_local nonnull ptr @_Z6getObjv() local_unnamed_addr {
entry:
  store ptr null, ptr getelementptr inbounds (%struct.Q, ptr @outer, i64 1, i32 0), align 8
  ret ptr @_ZL3Obj
}
