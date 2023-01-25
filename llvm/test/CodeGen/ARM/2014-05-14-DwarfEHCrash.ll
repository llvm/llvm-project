; Assertion `Encoding == DW_EH_PE_absptr && "Can handle absptr encoding only"' failed.
; Broken in r208166, fixed in 208715.

; RUN: llc -mtriple=arm-linux-androideabi -o - -filetype=asm -relocation-model=pic -simplifycfg-require-and-preserve-domtree=1 %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"
target triple = "armv4t--linux-androideabi"

@_ZTIi = external constant ptr

define void @_Z3fn2v() #0 personality ptr @__gxx_personality_v0 {
entry:
  invoke void @_Z3fn1v()
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %1 = extractvalue { ptr, i32 } %0, 1
  %2 = tail call i32 @llvm.eh.typeid.for(ptr @_ZTIi) #2
  %matches = icmp eq i32 %1, %2
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %lpad
  %3 = extractvalue { ptr, i32 } %0, 0
  %4 = tail call ptr @__cxa_begin_catch(ptr %3) #2
  tail call void @__cxa_end_catch() #2
  br label %try.cont

try.cont:                                         ; preds = %entry, %catch
  ret void

eh.resume:                                        ; preds = %lpad
  resume { ptr, i32 } %0
}

declare void @_Z3fn1v() #0

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(ptr) #1

declare ptr @__cxa_begin_catch(ptr)

declare void @__cxa_end_catch()

attributes #0 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
