; RUN: opt -passes=lcssa -S < %s | FileCheck %s

; This test is based on the following C++ code:
;
; void f()
; {
;   for (int i=0; i<12; i++) {
;     try {
;       if (i==3)
;         throw i;
;     } catch (int) {
;       continue;
;     } catch (...) { }
;     if (i==3) break;
;   }
; }
;
; The loop info analysis identifies the catch pad for the second catch as being
; outside the loop (because it returns to %for.end) but the associated
; catchswitch block is identified as being inside the loop.  Because of this
; analysis, the LCSSA pass wants to create a PHI node in the catchpad block
; for the catchswitch value, but this is a token, so it can't.

define void @f() personality ptr @__CxxFrameHandler3 {
entry:
  %tmp = alloca i32, align 4
  %i7 = alloca i32, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, 12
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %cond = icmp eq i32 %i.0, 3
  br i1 %cond, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  store i32 %i.0, ptr %tmp, align 4
  invoke void @_CxxThrowException(ptr %tmp, ptr nonnull @_TI1H) #1
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %if.then
  %tmp2 = catchswitch within none [label %catch, label %catch2] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %tmp3 = catchpad within %tmp2 [ptr @"\01??_R0H@8", i32 0, ptr %i7]
  catchret from %tmp3 to label %for.inc

catch2:                                           ; preds = %catch.dispatch
  %tmp4 = catchpad within %tmp2 [ptr null, i32 64, ptr null]
  catchret from %tmp4 to label %for.end

for.inc:                                          ; preds = %catch, %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %catch2, %for.cond
  ret void

unreachable:                                      ; preds = %if.then
  unreachable
}

; CHECK-LABEL: define void @f()
; CHECK: catch2:
; CHECK-NOT: phi
; CHECK:   %tmp4 = catchpad within %tmp2
; CHECK:   catchret from %tmp4 to label %for.end

%rtti.TypeDescriptor2 = type { ptr, ptr, [3 x i8] }
%eh.CatchableType = type { i32, i32, i32, i32, i32, i32, i32 }
%eh.CatchableTypeArray.1 = type { i32, [1 x i32] }
%eh.ThrowInfo = type { i32, i32, i32, i32 }

$"\01??_R0H@8" = comdat any

$"_CT??_R0H@84" = comdat any

$_CTA1H = comdat any

$_TI1H = comdat any

@"\01??_7type_info@@6B@" = external constant ptr
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { ptr @"\01??_7type_info@@6B@", ptr null, [3 x i8] c".H\00" }, comdat
@__ImageBase = external constant i8
@"_CT??_R0H@84" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R0H@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 4, i32 0 }, section ".xdata", comdat
@_CTA1H = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.1 { i32 1, [1 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0H@84" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32)] }, section ".xdata", comdat
@_TI1H = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @_CTA1H to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat

declare void @_CxxThrowException(ptr, ptr)

declare i32 @__CxxFrameHandler3(...)
