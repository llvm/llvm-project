; REQUIRES: x86_64-apple
; RUN: lli -jit-kind=orc %s
;
; Basic correctness testing for eh-frame processing and registration.

@_ZTIi = external constant ptr

declare ptr @__cxa_allocate_exception(i64)
declare void @__cxa_throw(ptr, ptr, ptr)

declare i32 @__gxx_personality_v0(...)
declare i32 @llvm.eh.typeid.for(ptr)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()

define void @explode() {
entry:
  %exception = tail call ptr @__cxa_allocate_exception(i64 4)
  store i32 42, ptr %exception, align 16
  tail call void @__cxa_throw(ptr %exception, ptr @_ZTIi, ptr null)
  unreachable
}

define i32 @main(i32 %argc, ptr %argv) personality ptr @__gxx_personality_v0 {
entry:
  invoke void @explode()
          to label %return unwind label %lpad

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %1 = extractvalue { ptr, i32 } %0, 1
  %2 = tail call i32 @llvm.eh.typeid.for(ptr @_ZTIi)
  %matches = icmp eq i32 %1, %2
  br i1 %matches, label %catch, label %eh.resume

catch:
  %3 = extractvalue { ptr, i32 } %0, 0
  %4 = tail call ptr @__cxa_begin_catch(ptr %3)
  %5 = load i32, ptr %4, align 4
  %cmp = icmp ne i32 %5, 42
  %cond = zext i1 %cmp to i32
  tail call void @__cxa_end_catch()
  br label %return

return:
  %retval.0 = phi i32 [ %cond, %catch ], [ 2, %entry ]
  ret i32 %retval.0

eh.resume:
  resume { ptr, i32 } %0
}
