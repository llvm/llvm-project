; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

@_ZTIc = external constant i8*
@_ZTId = external constant i8*
@_ZTIPKc = external constant i8*

define void @_Z3barv() uwtable optsize ssp personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @_Z3quxv() optsize
          to label %try.cont unwind label %lpad

try.cont:                                         ; preds = %entry, %invoke.cont4
  ret void

lpad:                                             ; preds = %entry
  %exn = landingpad {i8*, i32}
            cleanup
            catch i8** @_ZTIc
            filter [2 x i8**] [i8** @_ZTIPKc, i8** @_ZTId]
  resume { i8*, i32 } %exn
}

declare void @_Z3quxv() optsize

declare i32 @__gxx_personality_v0(...)

define void @cleanupret0() personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @_Z3quxv() optsize
          to label %exit unwind label %pad
pad:
  %cp = cleanuppad [i7 4]
  cleanupret %cp unwind to caller
exit:
  ret void
}

; forward ref by name
define void @cleanupret1() personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @_Z3quxv() optsize
          to label %exit unwind label %pad
cleanup:
  cleanupret %cp unwind label %pad
pad:
  %cp = cleanuppad []
  br label %cleanup
exit:
  ret void
}

; forward ref by ID
define void @cleanupret2() personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @_Z3quxv() optsize
          to label %exit unwind label %pad
cleanup:
  cleanupret %0 unwind label %pad
pad:
  %0 = cleanuppad []
  br label %cleanup
exit:
  ret void
}

define void @catchret0() personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @_Z3quxv() optsize
          to label %exit unwind label %pad
pad:
  %cp = catchpad [i7 4]
          to label %catch unwind label %endpad
catch:
  catchret %cp to label %exit
endpad:
  catchendpad unwind to caller
exit:
  ret void
}

; forward ref by name
define void @catchret1() personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @_Z3quxv() optsize
          to label %exit unwind label %pad
catch:
  catchret %cp to label %exit
pad:
  %cp = catchpad []
          to label %catch unwind label %endpad
endpad:
  catchendpad unwind to caller
exit:
  ret void
}

; forward ref by ID
define void @catchret2() personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @_Z3quxv() optsize
          to label %exit unwind label %pad
catch:
  catchret %0 to label %exit
pad:
  %0 = catchpad []
          to label %catch unwind label %endpad
endpad:
  catchendpad unwind to caller
exit:
  ret void
}

define i8 @catchpad() personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @_Z3quxv() optsize
          to label %exit unwind label %bb2
bb2:
  catchpad [i7 4] to label %exit unwind label %bb3
bb3:
  catchendpad unwind to caller
exit:
  ret i8 0
}

define void @terminatepad0() personality i32 (...)* @__gxx_personality_v0 {
entry:
  br label %try.cont

try.cont:
  invoke void @_Z3quxv() optsize
          to label %try.cont unwind label %bb
bb:
  terminatepad [i7 4] unwind label %bb
}

define void @terminatepad1() personality i32 (...)* @__gxx_personality_v0 {
entry:
  br label %try.cont

try.cont:
  invoke void @_Z3quxv() optsize
          to label %try.cont unwind label %bb
bb:
  terminatepad [i7 4] unwind to caller
}

define void @cleanuppad() personality i32 (...)* @__gxx_personality_v0 {
entry:
  br label %try.cont

try.cont:
  invoke void @_Z3quxv() optsize
          to label %try.cont unwind label %bb
bb:
  cleanuppad [i7 4]
  ret void
}

define void @catchendpad0() personality i32 (...)* @__gxx_personality_v0 {
entry:
  br label %try.cont

try.cont:
  invoke void @_Z3quxv() optsize
          to label %try.cont unwind label %bb
bb:
  catchendpad unwind label %bb
}

define void @catchendpad1() personality i32 (...)* @__gxx_personality_v0 {
entry:
  br label %try.cont

try.cont:
  invoke void @_Z3quxv() optsize
          to label %try.cont unwind label %bb
bb:
  catchendpad unwind to caller
}

define void @cleanupendpad0() personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @_Z3quxv() optsize
          to label %exit unwind label %pad
pad:
  %cp = cleanuppad [i7 4]
  invoke void @_Z3quxv() optsize
          to label %stop unwind label %endpad
stop:
  unreachable
endpad:
  cleanupendpad %cp unwind label %pad
exit:
  ret void
}

; forward ref by name
define void @cleanupendpad1() personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @_Z3quxv() optsize
          to label %exit unwind label %pad
endpad:
  cleanupendpad %cp unwind to caller
pad:
  %cp = cleanuppad []
  invoke void @_Z3quxv() optsize
          to label %stop unwind label %endpad
stop:
  unreachable
exit:
  ret void
}

; forward ref by ID
define void @cleanupendpad2() personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @_Z3quxv() optsize
          to label %exit unwind label %pad
endpad:
  cleanupendpad %0 unwind label %pad
pad:
  %0 = cleanuppad []
  invoke void @_Z3quxv() optsize
          to label %stop unwind label %endpad
stop:
  unreachable
exit:
  ret void
}
