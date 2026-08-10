; RUN: llc -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32-S64"
target triple = "armv7-none-linux-gnueabi"

@_ZTIi = external constant ptr

declare void @_Z3foov() noreturn;

declare ptr @__cxa_allocate_exception(i32)

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_throw(ptr, ptr, ptr)

declare void @__cxa_call_unexpected(ptr)

define i32 @main() personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: main:
entry:
  %exception.i = tail call ptr @__cxa_allocate_exception(i32 4) nounwind
  store i32 42, ptr %exception.i, align 4
  invoke void @__cxa_throw(ptr %exception.i, ptr @_ZTIi, ptr null) noreturn
          to label %unreachable.i unwind label %lpad.i

lpad.i:                                           ; preds = %entry
  %0 = landingpad { ptr, i32 }
          filter [1 x ptr] [ptr @_ZTIi]
          catch ptr @_ZTIi
; CHECK: .long	_ZTIi(target2)          @ TypeInfo 1
; CHECK: .long	_ZTIi(target2)          @ FilterInfo -1
  %1 = extractvalue { ptr, i32 } %0, 1
  %ehspec.fails.i = icmp slt i32 %1, 0
  br i1 %ehspec.fails.i, label %ehspec.unexpected.i, label %lpad.body

ehspec.unexpected.i:                              ; preds = %lpad.i
  %2 = extractvalue { ptr, i32 } %0, 0
  invoke void @__cxa_call_unexpected(ptr %2) noreturn
          to label %.noexc unwind label %lpad

.noexc:                                           ; preds = %ehspec.unexpected.i
  unreachable

unreachable.i:                                    ; preds = %entry
  unreachable

lpad:                                             ; preds = %ehspec.unexpected.i
  %3 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  br label %lpad.body

lpad.body:                                        ; preds = %lpad.i, %lpad
  %eh.lpad-body = phi { ptr, i32 } [ %3, %lpad ], [ %0, %lpad.i ]
  %4 = extractvalue { ptr, i32 } %eh.lpad-body, 1
  %5 = tail call i32 @llvm.eh.typeid.for(ptr @_ZTIi) nounwind
  %matches = icmp eq i32 %4, %5
  br i1 %matches, label %try.cont, label %eh.resume

try.cont:                                         ; preds = %lpad.body
  %6 = extractvalue { ptr, i32 } %eh.lpad-body, 0
  %7 = tail call ptr @__cxa_begin_catch(ptr %6) nounwind
  tail call void @__cxa_end_catch() nounwind
  ret i32 0

eh.resume:                                        ; preds = %lpad.body
  resume { ptr, i32 } %eh.lpad-body
}

declare i32 @llvm.eh.typeid.for(ptr) nounwind readnone

declare ptr @__cxa_begin_catch(ptr)

declare void @__cxa_end_catch()
