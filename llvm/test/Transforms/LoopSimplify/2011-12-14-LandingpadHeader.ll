; RUN: opt < %s -passes=loop-simplify -S | FileCheck %s
; PR11575

@catchtypeinfo = external unnamed_addr constant { i8*, i8*, i8* }

define void @main() uwtable ssp personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @f1()
          to label %try.cont19 unwind label %catch

; CHECK: catch.preheader:
; CHECK-NEXT: landingpad
; CHECK: br label %catch

; CHECK: catch.preheader.split-lp:
; CHECK-NEXT: landingpad
; CHECK: br label %catch

catch:                                            ; preds = %if.else, %entry
  %0 = landingpad { i8*, i32 }
          catch i8* bitcast ({ i8*, i8*, i8* }* @catchtypeinfo to i8*)
  invoke void @f3()
          to label %if.else unwind label %eh.resume

if.else:                                          ; preds = %catch
  invoke void @f2()
          to label %try.cont19 unwind label %catch

try.cont19:                                       ; preds = %if.else, %entry
  ret void

eh.resume:                                        ; preds = %catch
  %1 = landingpad { i8*, i32 }
          cleanup
          catch i8* bitcast ({ i8*, i8*, i8* }* @catchtypeinfo to i8*)
  resume { i8*, i32 } undef
}

declare i32 @__gxx_personality_v0(...)

declare void @f1()

declare void @f2()

declare void @f3()
