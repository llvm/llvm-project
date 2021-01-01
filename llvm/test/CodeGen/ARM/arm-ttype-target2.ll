; RUN: llc -mtriple=armv7-none-linux-gnueabi -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s

@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTS3Foo = linkonce_odr constant [5 x i8] c"3Foo\00"
@_ZTI3Foo = linkonce_odr unnamed_addr constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i32 2) to i8*), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @_ZTS3Foo, i32 0, i32 0) }

define i32 @main() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @_Z3foov()
          to label %return unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* bitcast ({ i8*, i8* }* @_ZTI3Foo to i8*)
  %1 = extractvalue { i8*, i32 } %0, 1
  %2 = tail call i32 @llvm.eh.typeid.for(i8* bitcast ({ i8*, i8* }* @_ZTI3Foo to i8*)) nounwind
; CHECK: _ZTI3Foo(target2)

  %matches = icmp eq i32 %1, %2
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %lpad
  %3 = extractvalue { i8*, i32 } %0, 0
  %4 = tail call i8* @__cxa_begin_catch(i8* %3) nounwind
  tail call void @__cxa_end_catch()
  br label %return

return:                                           ; preds = %entry, %catch
  %retval.0 = phi i32 [ 1, %catch ], [ 0, %entry ]
  ret i32 %retval.0

eh.resume:                                        ; preds = %lpad
  resume { i8*, i32 } %0
}

declare void @_Z3foov()

declare i32 @__gxx_personality_v0(...)

declare i32 @llvm.eh.typeid.for(i8*) nounwind readnone

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()
