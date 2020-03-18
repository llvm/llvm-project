; RUN: llc -mtriple armv7a--none-eabi   -enable-ipra=true -arm-extra-spills -arm-extra-spills-force -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple thumbv7a--none-eabi -enable-ipra=true -arm-extra-spills -arm-extra-spills-force -verify-machineinstrs < %s | FileCheck %s

; Test the interaction between IPRA and C++ exception handling. Currently, IPRA
; only marks registers as preserved on the non-exceptional return path, not in
; the landing pad.

declare dso_local i8* @__cxa_allocate_exception(i32) local_unnamed_addr
declare dso_local void @__cxa_throw(i8*, i8*, i8*) local_unnamed_addr
declare dso_local i32 @__gxx_personality_v0(...)
declare i32 @llvm.eh.typeid.for(i8*) nounwind readnone
declare dso_local i8* @__cxa_begin_catch(i8*) local_unnamed_addr
declare dso_local void @__cxa_end_catch() local_unnamed_addr

@g = dso_local local_unnamed_addr global i32 0, align 4
@_ZTIi = external dso_local constant i8*

define dso_local i32 @_Z11maybe_throwv() minsize {
; This function might return normally, or might throw an exception. r0 is used
; for a return value, we can preserve r1-r3 for IPRA.
; CHECK:      .save   {r1, r2, r3, lr}
; CHECK-NEXT: push    {r1, r2, r3, lr}
; CHECK:      pop{{(..)?}}    {r1, r2, r3, pc}
entry:
  %0 = load i32, i32* @g, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %exception = tail call i8* @__cxa_allocate_exception(i32 4)
  %1 = bitcast i8* %exception to i32*
  store i32 42, i32* %1, align 8
  tail call void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null)
  unreachable

if.else:                                          ; preds = %entry
  ret i32 1337
}

; Use inline assembly to force r0-r3 to be alive across a potentially throwing
; call, using them on the non-exceptional return path. r0 is the return value,
; so must be copied to another register. r1-r3 are voluntarily preserved by the
; callee, so can be left in those registers.
define dso_local i32 @_Z25test_non_exceptional_pathv() minsize personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK:      @APP
; CHECK-NEXT: @ def r0-r3
; CHECK-NEXT: @NO_APP
; CHECK-NEXT: mov     [[SAVE_R0:r[0-9]+]], r0
; CHECK-NEXT: .Ltmp{{.*}}
; CHECK-NEXT: bl      _Z11maybe_throwv
; CHECK:      mov     r0, [[SAVE_R0]]
; CHECK-NEXT: @APP
; CHECK-NEXT: @ use r0-r3
; CHECK-NEXT: @NO_APP
entry:
  %0 = tail call { i32, i32, i32, i32 } asm sideeffect "// def r0-r3", "={r0},={r1},={r2},={r3}"()
  %call = invoke i32 @_Z11maybe_throwv()
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 }
          cleanup
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %2 = extractvalue { i8*, i32 } %1, 1
  %3 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %2, %3
  br i1 %matches, label %catch, label %ehcleanup

catch:                                            ; preds = %lpad
  %4 = extractvalue { i8*, i32 } %1, 0
  %5 = tail call i8* @__cxa_begin_catch(i8* %4)
  %6 = bitcast i8* %5 to i32*
  %7 = load i32, i32* %6, align 4
  tail call void @__cxa_end_catch()
  br label %cleanup

try.cont:                                         ; preds = %entry
  %asmresult3 = extractvalue { i32, i32, i32, i32 } %0, 3
  %asmresult2 = extractvalue { i32, i32, i32, i32 } %0, 2
  %asmresult1 = extractvalue { i32, i32, i32, i32 } %0, 1
  %asmresult = extractvalue { i32, i32, i32, i32 } %0, 0
  tail call void asm sideeffect "// use r0-r3", "{r0},{r1},{r2},{r3}"(i32 %asmresult, i32 %asmresult1, i32 %asmresult2, i32 %asmresult3)
  br label %cleanup

cleanup:                                          ; preds = %try.cont, %catch
  %retval.0 = phi i32 [ 0, %try.cont ], [ %7, %catch ]
  ret i32 %retval.0

ehcleanup:                                        ; preds = %lpad
  resume { i8*, i32 } %1
}


; Use inline assembly to force r0-r3 to be alive across a potentially throwing
; call, using them after catching the exception. IPRA does not currently mark
; voluntarily preserved registers as live into the landing pad block, so all
; four registers must be copied elsewhere.
define dso_local i32 @_Z21test_exceptional_pathv() local_unnamed_addr minsize personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK:      @APP
; CHECK-NEXT: @ def r0-r3
; CHECK-NEXT: @NO_APP
; CHECK-DAG: mov [[SAVE_R0:r[0-9]+]], r0
; CHECK-DAG: mov [[SAVE_R1:r[0-9]+]], r1
; CHECK-DAG: mov [[SAVE_R2:r[0-9]+]], r2
; CHECK-DAG: mov [[SAVE_R3:r[0-9]+]], r3
; CHECK:      bl      _Z11maybe_throw

; CHECK:      bl      __cxa_begin_catch
; CHECK:      mov     r0, [[SAVE_R0]]
; CHECK-NEXT: mov     r1, [[SAVE_R1]]
; CHECK-NEXT: mov     r2, [[SAVE_R2]]
; CHECK-NEXT: mov     r3, [[SAVE_R3]]
; CHECK-NEXT: @APP
; CHECK-NEXT: @ use r0-r3
; CHECK-NEXT: @NO_APP
entry:
  %0 = tail call { i32, i32, i32, i32 } asm sideeffect "// def r0-r3", "={r0},={r1},={r2},={r3}"()
  %asmresult = extractvalue { i32, i32, i32, i32 } %0, 0
  %asmresult1 = extractvalue { i32, i32, i32, i32 } %0, 1
  %asmresult2 = extractvalue { i32, i32, i32, i32 } %0, 2
  %asmresult3 = extractvalue { i32, i32, i32, i32 } %0, 3
  %call = invoke i32 @_Z11maybe_throwv()
          to label %cleanup unwind label %lpad

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 }
          cleanup
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %2 = extractvalue { i8*, i32 } %1, 1
  %3 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %2, %3
  br i1 %matches, label %catch, label %ehcleanup

catch:                                            ; preds = %lpad
  %4 = extractvalue { i8*, i32 } %1, 0
  %5 = tail call i8* @__cxa_begin_catch(i8* %4)
  %6 = bitcast i8* %5 to i32*
  %7 = load i32, i32* %6, align 4
  tail call void asm sideeffect "// use r0-r3", "{r0},{r1},{r2},{r3}"(i32 %asmresult, i32 %asmresult1, i32 %asmresult2, i32 %asmresult3)
  tail call void @__cxa_end_catch()
  br label %cleanup

cleanup:                                          ; preds = %entry, %catch
  %retval.0 = phi i32 [ %7, %catch ], [ 0, %entry ]
  ret i32 %retval.0

ehcleanup:                                        ; preds = %lpad
  resume { i8*, i32 } %1
}
