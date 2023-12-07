; RUN: llc -mtriple=armv7-none-linux-gnueabi -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s

@_ZTVN10__cxxabiv117__class_type_infoE = external global ptr
@_ZTS3Foo = linkonce_odr constant [5 x i8] c"3Foo\00"
@_ZTI3Foo = linkonce_odr unnamed_addr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i32 2), ptr @_ZTS3Foo }

define i32 @main() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @_Z3foov()
          to label %return unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          catch ptr @_ZTI3Foo
  %1 = extractvalue { ptr, i32 } %0, 1
  %2 = tail call i32 @llvm.eh.typeid.for(ptr @_ZTI3Foo) nounwind
; CHECK: _ZTI3Foo(target2)

  %matches = icmp eq i32 %1, %2
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %lpad
  %3 = extractvalue { ptr, i32 } %0, 0
  %4 = tail call ptr @__cxa_begin_catch(ptr %3) nounwind
  tail call void @__cxa_end_catch()
  br label %return

return:                                           ; preds = %entry, %catch
  %retval.0 = phi i32 [ 1, %catch ], [ 0, %entry ]
  ret i32 %retval.0

eh.resume:                                        ; preds = %lpad
  resume { ptr, i32 } %0
}

declare void @_Z3foov()

declare i32 @__gxx_personality_v0(...)

declare i32 @llvm.eh.typeid.for(ptr) nounwind readnone

declare ptr @__cxa_begin_catch(ptr)

declare void @__cxa_end_catch()
