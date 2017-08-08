; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo

%C6object9ClassInfo = type { %C6object9ClassInfo__vtbl*, %C6object9ClassInfo* }
%C6object9ClassInfo__vtbl = type { %C6object9ClassInfo* }
%C6object9Exception__vtbl = type { %C6object9ClassInfo* }
%C6object6Object = type { %C6object6Object__vtbl* }
%C6object6Object__vtbl = type { %C6object9ClassInfo* }
%C6object9Throwable = type { %C6object9Throwable__vtbl* }
%C6object9Throwable__vtbl = type { %C6object9ClassInfo* }

@C6object9ClassInfo__ClassInfo = linkonce_odr constant %C6object9ClassInfo { %C6object9ClassInfo__vtbl* @C6object9ClassInfo__vtblZ, %C6object9ClassInfo* @C6object8TypeInfo__ClassInfo }
@C6object9ClassInfo__vtblZ = linkonce_odr constant %C6object9ClassInfo__vtbl { %C6object9ClassInfo* @C6object9ClassInfo__ClassInfo }
@C6object8TypeInfo__ClassInfo = linkonce_odr constant %C6object9ClassInfo { %C6object9ClassInfo__vtbl* @C6object9ClassInfo__vtblZ, %C6object9ClassInfo* @C6object6Object__ClassInfo }
@C6object6Object__ClassInfo = linkonce_odr constant %C6object9ClassInfo { %C6object9ClassInfo__vtbl* @C6object9ClassInfo__vtblZ, %C6object9ClassInfo* @C6object6Object__ClassInfo }
@C6object9Throwable__ClassInfo = linkonce_odr constant %C6object9ClassInfo { %C6object9ClassInfo__vtbl* @C6object9ClassInfo__vtblZ, %C6object9ClassInfo* @C6object6Object__ClassInfo }
@C6object9Exception__ClassInfo = linkonce_odr constant %C6object9ClassInfo { %C6object9ClassInfo__vtbl* @C6object9ClassInfo__vtblZ, %C6object9ClassInfo* @C6object9Throwable__ClassInfo }
@C6object9Exception__vtblZ = linkonce_odr constant %C6object9Exception__vtbl { %C6object9ClassInfo* @C6object9Exception__ClassInfo }
@C6object5Error__ClassInfo = linkonce_odr constant %C6object9ClassInfo { %C6object9ClassInfo__vtbl* @C6object9ClassInfo__vtblZ, %C6object9ClassInfo* @C6object9Throwable__ClassInfo }

define i32 @_D8test01494mainFMZi() personality i32 (i32, i32, i64, i8*, i8*)* @__sd_eh_personality {
body:
  %0 = invoke noalias i8* @_d_allocmemory(i64 8)
          to label %then unwind label %landingPad

then:                                             ; preds = %body
  %1 = bitcast i8* %0 to i8**
  store i8* bitcast (%C6object9Exception__vtbl* @C6object9Exception__vtblZ to i8*), i8** %1, align 8
  %2 = bitcast i8* %0 to %C6object6Object*
  invoke void @_D6object6Object6__ctorFMC6object6ObjectZv(%C6object6Object* %2)
          to label %then1 unwind label %landingPad

then1:                                            ; preds = %then
  %3 = bitcast i8* %0 to %C6object9Throwable*
  invoke void @__sd_eh_throw(%C6object9Throwable* nonnull %3)
          to label %then2 unwind label %landingPad

then2:                                            ; preds = %then1
  unreachable

landingPad:                                       ; preds = %then1, %then, %body
  %4 = landingpad { i8*, i32 }
          cleanup
          catch %C6object9ClassInfo* @C6object5Error__ClassInfo
          catch %C6object9ClassInfo* @C6object9Exception__ClassInfo
          catch %C6object9ClassInfo* @C6object9Throwable__ClassInfo
  %5 = extractvalue { i8*, i32 } %4, 1
  %6 = tail call i32 @llvm.eh.typeid.for(i8* nonnull bitcast (%C6object9ClassInfo* @C6object5Error__ClassInfo to i8*))
  %7 = icmp eq i32 %6, %5
  br i1 %7, label %catch, label %unwind3

catch:                                            ; preds = %unwind5, %unwind3, %landingPad
  %merge = phi i32 [ 23, %landingPad ], [ 19, %unwind3 ], [ 13, %unwind5 ]
  ret i32 %merge

unwind3:                                          ; preds = %landingPad
  %8 = tail call i32 @llvm.eh.typeid.for(i8* nonnull bitcast (%C6object9ClassInfo* @C6object9Exception__ClassInfo to i8*))
  %9 = icmp eq i32 %8, %5
  br i1 %9, label %catch, label %unwind5

unwind5:                                          ; preds = %unwind3
  %10 = tail call i32 @llvm.eh.typeid.for(i8* nonnull bitcast (%C6object9ClassInfo* @C6object9Throwable__ClassInfo to i8*))
  %11 = icmp eq i32 %10, %5
  br i1 %11, label %catch, label %unwind7

unwind7:                                          ; preds = %unwind5
  resume { i8*, i32 } %4
}

declare void @_D6object6Object6__ctorFMC6object6ObjectZv(%C6object6Object*)

declare noalias i8* @_d_allocmemory(i64)

declare i32 @__sd_eh_personality(i32, i32, i64, i8*, i8*)

declare void @__sd_eh_throw(%C6object9Throwable* nonnull) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #1

attributes #0 = { noreturn }
attributes #1 = { nounwind readnone }