; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo

%C6object9ClassInfo = type { ptr, ptr }
%C6object9ClassInfo__vtbl = type { ptr }
%C6object9Exception__vtbl = type { ptr }
%C6object6Object = type { ptr }
%C6object6Object__vtbl = type { ptr }
%C6object9Throwable = type { ptr }
%C6object9Throwable__vtbl = type { ptr }

@C6object9ClassInfo__ClassInfo = linkonce_odr constant %C6object9ClassInfo { ptr @C6object9ClassInfo__vtblZ, ptr @C6object8TypeInfo__ClassInfo }
@C6object9ClassInfo__vtblZ = linkonce_odr constant %C6object9ClassInfo__vtbl { ptr @C6object9ClassInfo__ClassInfo }
@C6object8TypeInfo__ClassInfo = linkonce_odr constant %C6object9ClassInfo { ptr @C6object9ClassInfo__vtblZ, ptr @C6object6Object__ClassInfo }
@C6object6Object__ClassInfo = linkonce_odr constant %C6object9ClassInfo { ptr @C6object9ClassInfo__vtblZ, ptr @C6object6Object__ClassInfo }
@C6object9Throwable__ClassInfo = linkonce_odr constant %C6object9ClassInfo { ptr @C6object9ClassInfo__vtblZ, ptr @C6object6Object__ClassInfo }
@C6object9Exception__ClassInfo = linkonce_odr constant %C6object9ClassInfo { ptr @C6object9ClassInfo__vtblZ, ptr @C6object9Throwable__ClassInfo }
@C6object9Exception__vtblZ = linkonce_odr constant %C6object9Exception__vtbl { ptr @C6object9Exception__ClassInfo }
@C6object5Error__ClassInfo = linkonce_odr constant %C6object9ClassInfo { ptr @C6object9ClassInfo__vtblZ, ptr @C6object9Throwable__ClassInfo }

define i32 @_D8test01494mainFMZi() personality ptr @__sd_eh_personality {
body:
  %0 = invoke noalias ptr @_d_allocmemory(i64 8)
          to label %then unwind label %landingPad

then:                                             ; preds = %body
  store ptr bitcast (ptr @C6object9Exception__vtblZ to ptr), ptr %0, align 8
  invoke void @_D6object6Object6__ctorFMC6object6ObjectZv(ptr %0)
          to label %then1 unwind label %landingPad

then1:                                            ; preds = %then
  invoke void @__sd_eh_throw(ptr nonnull %0)
          to label %then2 unwind label %landingPad

then2:                                            ; preds = %then1
  unreachable

landingPad:                                       ; preds = %then1, %then, %body
  %1 = landingpad { ptr, i32 }
          cleanup
          catch ptr @C6object5Error__ClassInfo
          catch ptr @C6object9Exception__ClassInfo
          catch ptr @C6object9Throwable__ClassInfo
  %2 = extractvalue { ptr, i32 } %1, 1
  %3 = tail call i32 @llvm.eh.typeid.for(ptr nonnull bitcast (ptr @C6object5Error__ClassInfo to ptr))
  %4 = icmp eq i32 %3, %2
  br i1 %4, label %catch, label %unwind3

catch:                                            ; preds = %unwind5, %unwind3, %landingPad
  %merge = phi i32 [ 23, %landingPad ], [ 19, %unwind3 ], [ 13, %unwind5 ]
  ret i32 %merge

unwind3:                                          ; preds = %landingPad
  %5 = tail call i32 @llvm.eh.typeid.for(ptr nonnull @C6object9Exception__ClassInfo)
  %6 = icmp eq i32 %5, %2
  br i1 %6, label %catch, label %unwind5

unwind5:                                          ; preds = %unwind3
  %7 = tail call i32 @llvm.eh.typeid.for(ptr nonnull @C6object9Throwable__ClassInfo)
  %8 = icmp eq i32 %7, %2
  br i1 %8, label %catch, label %unwind7

unwind7:                                          ; preds = %unwind5
  resume { ptr, i32 } %1
}

declare void @_D6object6Object6__ctorFMC6object6ObjectZv(ptr)

declare noalias ptr @_d_allocmemory(i64)

declare i32 @__sd_eh_personality(i32, i32, i64, ptr, ptr)

declare void @__sd_eh_throw(ptr nonnull) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(ptr) #1

attributes #0 = { noreturn }
attributes #1 = { nounwind readnone }
