; RUN: llc < %s | FileCheck %s
; RUN: llc < %s -verify-machineinstrs -stop-after=x86-avoid-trailing-call -o - | FileCheck %s --check-prefix=MIR

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.28.29914"

@str = private unnamed_addr constant [6 x i8] c"Boom!\00", align 1

define dso_local void @trap() {
entry:
  unreachable
}

define dso_local void @test() personality ptr @__CxxFrameHandler3 {
entry:

; Windows x64 unwinding may insert padding after a call to ensure that the
; return address is in the same IP2State as the call instruction itself. We also
; do the same for inline asm that may unwind, as we don't know if there is
; call instruction inside that asm.
; CHECK-LABEL: .Ltmp0:
; CHECK: callq  trap
; CHECK: nop
; CHECK-LABEL: .Ltmp1:
; MIR-LABEL: name: test
; MIR:       INLINEASM &"call trap", sideeffect unwind
; MIR-NEXT:  NOOP
; MIR-NEXT:  EH_LABEL

  invoke void asm sideeffect unwind "call trap", "~{dirflag},~{fpsr},~{flags}"()
          to label %exit unwind label %except

exit:
  ret void

except:

; CHECK-LABEL: "?dtor$2@?0?test@4HA":
; CHECK: callq	printf
; CHECK: nop

  %0 = cleanuppad within none []
  call void (ptr, ...) @printf(ptr @str) [ "funclet"(token %0) ]
  cleanupret from %0 unwind to caller
}

declare dso_local i32 @__CxxFrameHandler3(...)

declare dso_local void @printf(ptr, ...)

; SEH Table

; CHECK-LABEL: $cppxdata$test:
; CHECK-NEXT:    .long    429065506                       # MagicNumber
; CHECK-NEXT:    .long    1                               # MaxState
; CHECK-NEXT:    .long    $stateUnwindMap$test@IMGREL   # UnwindMap
; CHECK-NEXT:    .long    0                               # NumTryBlocks
; CHECK-NEXT:    .long    0                               # TryBlockMap
; CHECK-NEXT:    .long    3                               # IPMapEntries
; CHECK-NEXT:    .long    $ip2state$test@IMGREL         # IPToStateXData
; CHECK-NEXT:    .long    40                              # UnwindHelp
; CHECK-NEXT:    .long    0                               # ESTypeList
; CHECK-NEXT:    .long    1                               # EHFlags
; CHECK-NEXT:$stateUnwindMap$test:
; CHECK-NEXT:    .long    -1                              # ToState
; CHECK-NEXT:    .long    "?dtor$2@?0?test@4HA"@IMGREL    # Action
; CHECK-NEXT:$ip2state$test:
; CHECK-NEXT:    .long    .Lfunc_begin0@IMGREL            # IP
; CHECK-NEXT:    .long    -1                              # ToState
; CHECK-NEXT:    .long    .Ltmp0@IMGREL                   # IP
; CHECK-NEXT:    .long    0                               # ToState
; CHECK-NEXT:    .long    .Ltmp1@IMGREL                   # IP
; CHECK-NEXT:    .long    -1                              # ToState

; Empty unwinding asm has no call return address to pad.
define dso_local void @test_empty() personality ptr @__CxxFrameHandler3 {
; CHECK-LABEL: test_empty:
; CHECK:       #APP
; CHECK-NEXT:  #NO_APP
; CHECK-NEXT:  .Ltmp{{[0-9]+}}:{{.*}}# EH_LABEL
; MIR-LABEL:   name: test_empty
; MIR:         INLINEASM &"", sideeffect unwind
; MIR-NEXT:    EH_LABEL
entry:
  invoke void asm sideeffect unwind "", ""()
          to label %exit unwind label %except

exit:
  ret void

except:
  %cp = cleanuppad within none []
  call void (ptr, ...) @printf(ptr @str) [ "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller
}
