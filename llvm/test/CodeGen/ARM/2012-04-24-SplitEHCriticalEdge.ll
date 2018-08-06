; RUN: llc -mtriple=thumbv7-apple-ios -relocation-model=pic -disable-fp-elim -mcpu=cortex-a8 < %s

; CodeGen SplitCriticalEdge() shouldn't try to break edge to a landing pad.
; rdar://11300144

%0 = type opaque
%class.FunctionInterpreter.3.15.31 = type { %class.Parser.1.13.29, %class.Parser.1.13.29*, %struct.ParserVariable.2.14.30*, i32 }
%class.Parser.1.13.29 = type { i32 (...)**, %class.Parser.1.13.29* }
%struct.ParserVariable.2.14.30 = type opaque
%struct.ParseErrorMsg.0.12.28 = type { i32, i32, i32 }

@_ZTI13ParseErrorMsg = external hidden unnamed_addr constant { i8*, i8* }
@"OBJC_IVAR_$_MUMathExpressionDoubleBased.mInterpreter" = external hidden global i32, section "__DATA, __objc_ivar", align 4
@"\01L_OBJC_SELECTOR_REFERENCES_14" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"

declare i8* @objc_msgSend(i8*, i8*, ...)

declare i32 @llvm.eh.typeid.for(i8*) nounwind readnone

declare i8* @__cxa_begin_catch(i8*)

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind

declare void @__cxa_end_catch()

declare void @_ZSt9terminatev()

define hidden double @t(%0* %self, i8* nocapture %_cmd) optsize ssp personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*) {
entry:
  %call = invoke double undef(%class.FunctionInterpreter.3.15.31* undef) optsize
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* bitcast ({ i8*, i8* }* @_ZTI13ParseErrorMsg to i8*)
  br i1 undef, label %catch, label %eh.resume

catch:                                            ; preds = %lpad
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, %struct.ParseErrorMsg.0.12.28*)*)(i8* undef, i8* undef, %struct.ParseErrorMsg.0.12.28* undef) optsize
          to label %invoke.cont2 unwind label %lpad1

invoke.cont2:                                     ; preds = %catch
  br label %try.cont

try.cont:                                         ; preds = %invoke.cont2, %entry
  %value.0 = phi double [ 0x7FF8000000000000, %invoke.cont2 ], [ %call, %entry ]
  ret double %value.0

lpad1:                                            ; preds = %catch
  %1 = landingpad { i8*, i32 }
          cleanup
  invoke void @__cxa_end_catch()
          to label %eh.resume unwind label %terminate.lpad

eh.resume:                                        ; preds = %lpad1, %lpad
  resume { i8*, i32 } undef

terminate.lpad:                                   ; preds = %lpad1
  %2 = landingpad { i8*, i32 }
          catch i8* null
  unreachable
}

declare i32 @__gxx_personality_sj0(...)

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"Objective-C Version", i32 2}
!1 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!2 = !{i32 1, !"Objective-C Image Info Section", !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!3 = !{i32 4, !"Objective-C Garbage Collection", i32 0}
