; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.28.29914"

@str = private unnamed_addr constant [6 x i8] c"Boom!\00", align 1

define dso_local void @trap() {
entry:
  unreachable
}

define dso_local void @test() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:

; CHECK-NOT: .Ltmp0:
; CHECK: callq  trap
; CHECK-NOT: .Ltmp1:

  invoke void asm sideeffect "call trap", "~{dirflag},~{fpsr},~{flags}"()
          to label %exit unwind label %except

exit:
  ret void

except:

; CHECK-LABEL: "?dtor$2@?0?test@4HA":
; CHECK: callq	printf

  %0 = cleanuppad within none []
  call void (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @str, i64 0, i64 0)) [ "funclet"(token %0) ]
  cleanupret from %0 unwind to caller
}

declare dso_local i32 @__CxxFrameHandler3(...)

declare dso_local void @printf(i8*, ...)

; SEH Table

; CHECK-LABEL: $cppxdata$test:
; CHECK-NEXT:    .long    429065506                       # MagicNumber
; CHECK-NEXT:    .long    1                               # MaxState
; CHECK-NEXT:    .long    ($stateUnwindMap$test)@IMGREL   # UnwindMap
; CHECK-NEXT:    .long    0                               # NumTryBlocks
; CHECK-NEXT:    .long    0                               # TryBlockMap
; CHECK-NEXT:    .long    1                               # IPMapEntries
; CHECK-NEXT:    .long    ($ip2state$test)@IMGREL         # IPToStateXData
; CHECK-NEXT:    .long    40                              # UnwindHelp
; CHECK-NEXT:    .long    0                               # ESTypeList
; CHECK-NEXT:    .long    1                               # EHFlags
; CHECK-NEXT:$stateUnwindMap$test:
; CHECK-NEXT:    .long    -1                              # ToState
; CHECK-NEXT:    .long    "?dtor$2@?0?test@4HA"@IMGREL    # Action
; CHECK-NEXT:$ip2state$test:
; CHECK-NEXT:    .long    .Lfunc_begin0@IMGREL            # IP
; CHECK-NEXT:    .long    -1                              # ToState
