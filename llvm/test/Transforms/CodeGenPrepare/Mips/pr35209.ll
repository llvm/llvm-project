; RUN: opt -S -mtriple=mips64-mti-linux-gnu -codegenprepare < %s | FileCheck %s

; Test that if an address that was sunk from a dominating bb, used in a
; select that is erased along with its' trivally dead operand, that the
; sunken address is not reused if the same address computation occurs
; after the select. Previously, this caused a ICE.

%struct.az = type { i32, ptr }
%struct.bt = type { i32 }
%struct.f = type { %struct.ax, %union.anon }
%struct.ax = type { ptr }
%union.anon = type { %struct.bd }
%struct.bd = type { i64 }
%struct.bg = type { i32, i32 }
%struct.ap = type { i32, i32 }

@ch = common global %struct.f zeroinitializer, align 8
@j = common global ptr null, align 8
@ck = common global i32 0, align 4
@h = common global i32 0, align 4
@.str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1

define internal void @probestart() {
entry:
  %load0 = load ptr, ptr @j, align 8
  %bw = getelementptr inbounds %struct.az, ptr %load0, i64 0, i32 1
  %load1 = load i32, ptr @h, align 4
  %cond = icmp eq i32 %load1, 0
  br i1 %cond, label %sw.bb, label %cl

sw.bb:                                            ; preds = %entry
  %call = tail call inreg { i64, i64 } @ba(ptr @ch)
  br label %cl

cl:                                               ; preds = %sw.bb, %entry
  %load2 = load ptr, ptr %bw, align 8
  %tobool = icmp eq ptr %load2, null
  %load3 = load i32, ptr @ck, align 4
  %.sink5 = select i1 %tobool, ptr getelementptr (%struct.bg, ptr getelementptr inbounds (%struct.f, ptr @ch, i64 0, i32 1), i64 0, i32 1), ptr getelementptr (%struct.ap, ptr getelementptr inbounds (%struct.f, ptr @ch, i64 0, i32 1), i64 0, i32 1)
  store i32 %load3, ptr %.sink5, align 4
  store i32 1, ptr getelementptr inbounds (%struct.f, ptr @ch, i64 0, i32 1, i32 0, i32 0), align 8
  %load4 = load ptr, ptr %bw, align 8
  tail call void (ptr, ...) @a(ptr @.str, ptr %load4)
  ret void
}

; CHECK-LABEL: @probestart()
; CHECK-LABEL: entry:
; CHECK: %[[I0:[a-z0-9]+]] = load ptr, ptr @j
; CHECK-LABEL: cl:

; CHECK-NOT: %{{[a-z0-9]+}}  = load ptr, ptr %bw
; CHECK-NOT: %{{[.a-z0-9]}} = select
; CHECK-NOT: %{{[a-z0-9]+}}  = load ptr, ptr %bw

; CHECK: %sunkaddr = getelementptr inbounds i8, ptr %[[I0]], i64 8
; CHECK-NEXT: %{{[a-z0-9]+}} = load ptr, ptr %sunkaddr
; CHECK-NEXT: tail call void (ptr, ...) @a

declare inreg { i64, i64 } @ba(ptr)

declare void @a(ptr, ...)
