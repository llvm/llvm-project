; RUN: llc < %s
; PR7716
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10.0.0"

%0 = type { ptr, ptr }
%struct.A = type { i32 }

@d = internal global i32 0, align 4               ; <ptr> [#uses=6]
@_ZTVN10__cxxabiv117__class_type_infoE = external global ptr ; <ptr> [#uses=1]
@_ZTS1A = internal constant [3 x i8] c"1A\00"     ; <ptr> [#uses=1]
@_ZTI1A = internal constant %0 { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i32 2), ptr @_ZTS1A } ; <ptr> [#uses=1]
@.str2 = private constant [18 x i8] c"c == %d, d == %d\0A\00" ; <ptr> [#uses=1]
@.str3 = private constant [16 x i8] c"A(const A&) %d\0A\00" ; <ptr> [#uses=1]
@.str4 = private constant [9 x i8] c"~A() %d\0A\00" ; <ptr> [#uses=1]
@.str5 = private constant [8 x i8] c"A() %d\0A\00" ; <ptr> [#uses=1]
@str = internal constant [14 x i8] c"Throwing 1...\00" ; <ptr> [#uses=1]
@str1 = internal constant [8 x i8] c"Caught.\00"  ; <ptr> [#uses=1]

declare i32 @printf(ptr nocapture, ...) nounwind

declare ptr @__cxa_allocate_exception(i32)

declare i32 @__gxx_personality_sj0(...)

declare i32 @llvm.eh.typeid.for(ptr) nounwind

declare void @_Unwind_SjLj_Resume(ptr)

define internal void @_ZN1AD1Ev(ptr nocapture %this) nounwind ssp align 2 {
entry:
  %tmp.i = getelementptr inbounds %struct.A, ptr %this, i32 0, i32 0 ; <ptr> [#uses=1]
  %tmp2.i = load i32, ptr %tmp.i                      ; <i32> [#uses=1]
  %call.i = tail call i32 (ptr, ...) @printf(ptr @.str4, i32 %tmp2.i) nounwind ; <i32> [#uses=0]
  %tmp3.i = load i32, ptr @d                          ; <i32> [#uses=1]
  %inc.i = add nsw i32 %tmp3.i, 1                 ; <i32> [#uses=1]
  store i32 %inc.i, ptr @d
  ret void
}

declare void @__cxa_throw(ptr, ptr, ptr)

define i32 @main() ssp personality ptr @__gxx_personality_sj0 {
entry:
  %puts.i = tail call i32 @puts(ptr @str) ; <i32> [#uses=0]
  %exception.i = tail call ptr @__cxa_allocate_exception(i32 4) nounwind ; <ptr> [#uses=2]
  store i32 1, ptr %exception.i
  %call.i.i.i = tail call i32 (ptr, ...) @printf(ptr @.str5, i32 1) nounwind ; <i32> [#uses=0]
  invoke void @__cxa_throw(ptr %exception.i, ptr @_ZTI1A, ptr @_ZN1AD1Ev) noreturn
          to label %.noexc unwind label %lpad

.noexc:                                           ; preds = %entry
  unreachable

try.cont:                                         ; preds = %lpad
  %0 = tail call ptr @__cxa_get_exception_ptr(ptr %exn) nounwind ; <ptr> [#uses=0]
  %call.i.i = tail call i32 (ptr, ...) @printf(ptr @.str3, i32 2) nounwind ; <i32> [#uses=0]
  %1 = tail call ptr @__cxa_begin_catch(ptr %exn) nounwind ; <ptr> [#uses=0]
  %puts = tail call i32 @puts(ptr @str1) ; <i32> [#uses=0]
  %call.i.i3 = tail call i32 (ptr, ...) @printf(ptr @.str4, i32 2) nounwind ; <i32> [#uses=0]
  %tmp3.i.i = load i32, ptr @d                        ; <i32> [#uses=1]
  %inc.i.i4 = add nsw i32 %tmp3.i.i, 1            ; <i32> [#uses=1]
  store i32 %inc.i.i4, ptr @d
  tail call void @__cxa_end_catch()
  %tmp13 = load i32, ptr @d                           ; <i32> [#uses=1]
  %call14 = tail call i32 (ptr, ...) @printf(ptr @.str2, i32 2, i32 %tmp13) ; <i32> [#uses=0]
  %tmp16 = load i32, ptr @d                           ; <i32> [#uses=1]
  %cmp = icmp ne i32 %tmp16, 2                    ; <i1> [#uses=1]
  %conv = zext i1 %cmp to i32                     ; <i32> [#uses=1]
  ret i32 %conv

lpad:                                             ; preds = %entry
  %exn.ptr = landingpad { ptr, i32 }
           catch ptr @_ZTI1A
           catch ptr null
  %exn = extractvalue { ptr, i32 } %exn.ptr, 0
  %eh.selector = extractvalue { ptr, i32 } %exn.ptr, 1
  %2 = tail call i32 @llvm.eh.typeid.for(ptr @_ZTI1A) nounwind ; <i32> [#uses=1]
  %3 = icmp eq i32 %eh.selector, %2               ; <i1> [#uses=1]
  br i1 %3, label %try.cont, label %eh.resume

eh.resume:                                        ; preds = %lpad
  tail call void @_Unwind_SjLj_Resume(ptr %exn) noreturn
  unreachable
}

declare ptr @__cxa_get_exception_ptr(ptr)

declare ptr @__cxa_begin_catch(ptr)

declare void @__cxa_end_catch()

declare i32 @puts(ptr nocapture) nounwind
