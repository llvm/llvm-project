; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu -frame-pointer=all | FileCheck %s

define i32 @_Z4funci(i32 %a) ssp {
; CHECK:       mflr 0
; CHECK-NEXT:  stwu 1, -32(1)
; CHECK-NEXT:  stw 31, 28(1)
; CHECK-NEXT:  stw 0, 36(1)
; CHECK:  mr 31, 1
entry:
  %a_addr = alloca i32                            ; <ptr> [#uses=2]
  %retval = alloca i32                            ; <ptr> [#uses=2]
  %0 = alloca i32                                 ; <ptr> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store i32 %a, ptr %a_addr
  %1 = call i32 @_Z3barPi(ptr %a_addr)           ; <i32> [#uses=1]
  store i32 %1, ptr %0, align 4
  %2 = load i32, ptr %0, align 4                      ; <i32> [#uses=1]
  store i32 %2, ptr %retval, align 4
  br label %return

return:                                           ; preds = %entry
  %retval1 = load i32, ptr %retval                    ; <i32> [#uses=1]
  ret i32 %retval1
}

declare i32 @_Z3barPi(ptr)
