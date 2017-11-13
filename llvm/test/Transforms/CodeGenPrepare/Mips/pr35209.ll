; RUN: opt -S -mtriple=mips64-mti-linux-gnu -codegenprepare < %s | FileCheck %s

; Test that if an address that was sunk from a dominating bb, used in a
; select that is erased along with its' trivally dead operand, that the
; sunken address is not reused if the same address computation occurs
; after the select. Previously, this caused a ICE.

%struct.az = type { i32, %struct.bt* }
%struct.bt = type { i32 }
%struct.f = type { %struct.ax, %union.anon }
%struct.ax = type { %struct.az* }
%union.anon = type { %struct.bd }
%struct.bd = type { i64 }
%struct.bg = type { i32, i32 }
%struct.ap = type { i32, i32 }

@ch = common global %struct.f zeroinitializer, align 8
@j = common global %struct.az* null, align 8
@ck = common global i32 0, align 4
@h = common global i32 0, align 4
@.str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1

define internal void @probestart() {
entry:
  %0 = load %struct.az*, %struct.az** @j, align 8
  %bw = getelementptr inbounds %struct.az, %struct.az* %0, i64 0, i32 1
  %1 = load i32, i32* @h, align 4
  %cond = icmp eq i32 %1, 0
  br i1 %cond, label %sw.bb, label %cl

sw.bb:                                            ; preds = %entry
  %call = tail call inreg { i64, i64 } @ba(i32* bitcast (%struct.f* @ch to i32*))
  br label %cl

cl:                                               ; preds = %sw.bb, %entry
  %2 = load %struct.bt*, %struct.bt** %bw, align 8
  %tobool = icmp eq %struct.bt* %2, null
  %3 = load i32, i32* @ck, align 4
  %.sink5 = select i1 %tobool, i32* getelementptr (%struct.bg, %struct.bg* bitcast (%union.anon* getelementptr inbounds (%struct.f, %struct.f* @ch, i64 0, i32 1) to %struct.bg*), i64 0, i32 1), i32* getelementptr (%struct.ap, %struct.ap* bitcast (%union.anon* getelementptr inbounds (%struct.f, %struct.f* @ch, i64 0, i32 1) to %struct.ap*), i64 0, i32 1)
  store i32 %3, i32* %.sink5, align 4
  store i32 1, i32* bitcast (i64* getelementptr inbounds (%struct.f, %struct.f* @ch, i64 0, i32 1, i32 0, i32 0) to i32*), align 8
  %4 = load %struct.bt*, %struct.bt** %bw, align 8
  tail call void (i8*, ...) @a(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i64 0, i64 0), %struct.bt* %4)
  ret void
}

; CHECK-LABEL: @probestart()
; CHECK-LABEL: entry:
; CHECK: %[[I0:[0-9]+]] = load %struct.az*, %struct.az** @j
; CHECK-LABEL: cl:

; CHECK-NOT: %{{[0-9]+}}  = load %struct.bt*, %struct.bt** %bw
; CHECK-NOT: %{{[.a-z0-9]}} = select
; CHECK-NOT: %{{[0-9]+}}  = load %struct.bt*, %struct.bt** %bw

; CHECK: %[[I1:[0-9]+]] = bitcast %struct.az* %[[I0]] to i8*
; CHECK-NEXT: %sunkaddr = getelementptr i8, i8* %[[I1]], i64 8
; CHECK-NEXT: %[[I2:[0-9]+]] = bitcast i8* %sunkaddr to %struct.bt**
; CHECK-NEXT: %{{[0-9]+}} = load %struct.bt*, %struct.bt** %[[I2]]
; CHECK-NEXT: tail call void (i8*, ...) @a

declare inreg { i64, i64 } @ba(i32*)

declare void @a(i8*, ...)
