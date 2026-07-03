; REQUIRES: system-darwin && host-unwind-supports-jit
; RUN: lli -jit-kind=orc %s
;
; Check that we can throw exceptions from no-fp functions. On systems that
; support compact-unwind this implicitly tests that we correctly handle
; unwind-info records that depend on DWARF FDEs.

@_ZTIi = external constant ptr

declare ptr @__cxa_allocate_exception(i64)
declare void @__cxa_throw(ptr, ptr, ptr)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()
declare i32 @__gxx_personality_v0(...)
declare i32 @llvm.eh.typeid.for.p0(ptr)

define void @_Z3foov() "frame-pointer"="none" {
entry:
  %exception = tail call ptr @__cxa_allocate_exception(i64 4)
  store i32 42, ptr %exception
  tail call void @__cxa_throw(ptr %exception, ptr nonnull @_ZTIi, ptr null)
  unreachable
}

define i32 @main(i32 %argc, ptr %argv) "frame-pointer"="all" personality ptr @__gxx_personality_v0 {
entry:
  invoke void @_Z3foov()
          to label %return.unreachable unwind label %lpad

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %1 = extractvalue { ptr, i32 } %0, 1
  %2 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIi)
  %matches = icmp eq i32 %1, %2
  br i1 %matches, label %catch, label %eh.resume

catch:
  %3 = extractvalue { ptr, i32 } %0, 0
  %4 = tail call ptr @__cxa_begin_catch(ptr %3)
  %5 = load i32, ptr %4
  %sub = sub nsw i32 42, %5
  tail call void @__cxa_end_catch()
  ret i32 %sub

return.unreachable:
  unreachable

eh.resume:
  resume { ptr, i32 } %0
}
