; RUN: llc < %s | FileCheck %s
; CHECK: .long   .Ltmp6-.Ltmp12          #   Call between .Ltmp12 and .Ltmp6

; We used to crash in filetype=obj when computing a negative value.
; RUN: llc -filetype=obj < %s

target triple = "powerpc--netbsd"
@_ZTI1I = external constant { i8*, i8* }
define void @f(i8 %foo, i32 %bar) personality i8* bitcast (void ()* @g to i8*) {
  invoke void @g()
          to label %try.cont unwind label %lpad
lpad:                                             ; preds = %0
  %tmp = landingpad { i8*, i32 }
          catch i8* bitcast ({ i8*, i8* }* @_ZTI1I to i8*)
  br i1 undef, label %catch10, label %catch
catch10:                                          ; preds = %lpad
  %tmp8 = load i32, i32* undef, align 4
  %conv.i.i = zext i8 %foo to i32
  %cond.i.i = select i1 undef, i32 %conv.i.i, i32 %tmp8
  invoke void @_Z24__put_character_sequenceIccEvR1AIT_T0_Ej(i32 %cond.i.i)
          to label %invoke.cont20 unwind label %lpad15
invoke.cont20:                                    ; preds = %catch10
  ret void
try.cont:                                         ; preds = %0
  ret void
catch:                                            ; preds = %lpad
  %tmp14 = load i32, i32* undef, align 4
  %conv.i.i34 = zext i8 %foo to i32
  %cond.i.i35 = select i1 undef, i32 %conv.i.i34, i32 %tmp14
  invoke void @_Z24__put_character_sequenceIccEvR1AIT_T0_Ej(i32 %cond.i.i35)
          to label %invoke.cont8 unwind label %lpad3
invoke.cont8:                                     ; preds = %call2.i.i.noexc36
  ret void
lpad3:                                            ; preds = %call2.i.i.noexc36, %catch
  %tmp16 = landingpad { i8*, i32 }
          cleanup
  invoke void @g()
          to label %eh.resume unwind label %terminate.lpad
lpad15:                                           ; preds = %catch10
  %tmp19 = landingpad { i8*, i32 }
          cleanup
  invoke void @g()
          to label %eh.resume unwind label %terminate.lpad
eh.resume:                                        ; preds = %lpad15, %lpad3
  ret void
terminate.lpad:                                   ; preds = %lpad15, %lpad3
  %tmp22 = landingpad { i8*, i32 }
          catch i8* null
  ret void
}
declare void @g()
declare void @_Z24__put_character_sequenceIccEvR1AIT_T0_Ej(i32)
