; RUN: llc -mtriple armv7-apple-ios -relocation-model=pic -o - %s | FileCheck %s -check-prefix=ARM-PIC -check-prefix=ARM
; RUN: llc -mtriple armv7-apple-ios -relocation-model=static -o - %s | FileCheck %s -check-prefix=ARM-NOPIC -check-prefix=ARM
; RUN: llc -mtriple armv7-apple-ios -relocation-model=dynamic-no-pic -o - %s | FileCheck %s -check-prefix=ARM-NOPIC -check-prefix=ARM
; RUN: llc -mtriple thumbv6-apple-ios -relocation-model=pic -o - %s | FileCheck %s -check-prefix=THUMB1-PIC -check-prefix=THUMB1
; RUN: llc -mtriple thumbv6-apple-ios -relocation-model=static -o - %s | FileCheck %s -check-prefix=THUMB1-NOPIC -check-prefix=THUMB1
; RUN: llc -mtriple thumbv6-apple-ios -relocation-model=dynamic-no-pic -o - %s | FileCheck %s -check-prefix=THUMB1-NOPIC -check-prefix=THUMB1

@_ZTIi = external constant i8*

define i32 @main() #0 personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*) {
entry:
  %exception = tail call i8* @__cxa_allocate_exception(i32 4) #1
  %0 = bitcast i8* %exception to i32*
  store i32 1, i32* %0, align 4
  invoke void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null) #2
          to label %unreachable unwind label %lpad

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 }
          catch i8* null
  %2 = extractvalue { i8*, i32 } %1, 0
  %3 = tail call i8* @__cxa_begin_catch(i8* %2) #1
  tail call void @__cxa_end_catch()
  ret i32 0

unreachable:                                      ; preds = %entry
  unreachable
}

declare i8* @__cxa_allocate_exception(i32)

declare void @__cxa_throw(i8*, i8*, i8*)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

declare i32 @__gxx_personality_sj0(...)

attributes #0 = { ssp }
attributes #1 = { nounwind }
attributes #2 = { noreturn }

; ARM: vst1.64
; ARM: vst1.64

; ARM-PIC: cxa_throw
; ARM-PIC: trap
; ARM-PIC: adr [[REG1:r[0-9]+]], [[LJTI:.*]]
; ARM-PIC: ldr [[REG0:r[0-9]+]], [r{{[0-9]+}}, [[REG1]]]
; ARM-PIC: add pc, [[REG0]], [[REG1]]
; ARM-PIC: [[LJTI]]
; ARM-PIC: .data_region jt32
; ARM-PIC: .long [[LABEL:LBB0_[0-9]]]-[[LJTI]]
; ARM-PIC: .end_data_region
; ARM-PIC: [[LABEL]]

; ARM-NOPIC: cxa_throw
; ARM-NOPIC: trap
; ARM-NOPIC: adr [[REG1:r[0-9]+]], [[LJTI:.*]]
; ARM-NOPIC: ldr [[REG0:r[0-9]+]], [r{{[0-9]+}}, [[REG1]]]
; ARM-NOPIC: mov pc, [[REG0]]
; ARM-NOPIC: [[LJTI]]
; ARM-NOPIC: .data_region jt32
; ARM-NOPIC: .long [[LABEL:LBB0_[0-9]]]
; ARM-NOPIC: .end_data_region
; ARM-NOPIC: [[LABEL]]

; ARM: vld1.64
; ARM: vld1.64

; On Thumb1 targets, we have no way to preserve the floating-point registers.
; If all other code is built for Thumb1 or is built soft-float, this is not a
; problem as the FP regs don't need saving. However, if this code is linked
; against ARM code that uses the FP regs, they won't be restored correctly. We
; don't support this use-case, but have no way to prevent it in the compiler.

; THUMB1: push {{[^d]*$}}
; THUMB1-NOT: vst

; THUMB1-PIC: cxa_throw
; THUMB1-PIC: trap
; THUMB1-PIC: adr [[REG1:r[0-9]+]], [[LJTI:.*]]
; THUMB1-PIC: adds [[REG0:r[0-9]+]], [[REG0]], [[REG1]]
; THUMB1-PIC: ldr [[REG0]]
; THUMB1-PIC: adds [[REG0]], [[REG0]], [[REG1]]
; THUMB1-PIC: mov pc, [[REG0]]
; THUMB1-PIC: [[LJTI]]
; THUMB1-PIC: .data_region jt32
; THUMB1-PIC: .long [[LABEL:LBB0_[0-9]]]-[[LJTI]]
; THUMB1-PIC: .end_data_region
; THUMB1-PIC: [[LABEL]]

; THUMB1-NOPIC: cxa_throw
; THUMB1-NOPIC: trap
; THUMB1-NOPIC: adr [[REG1:r[0-9]+]], [[LJTI:.*]]
; THUMB1-NOPIC: adds [[REG0:r[0-9]+]], [[REG0]], [[REG1]]
; THUMB1-NOPIC: ldr [[REG0]]
; THUMB1-NOPIC: mov pc, [[REG0]]
; THUMB1-NOPIC: [[LJTI]]
; THUMB1-NOPIC: .data_region jt32
; THUMB1-NOPIC: .long [[LABEL:LBB0_[0-9]]]+1
; THUMB1-NOPIC: .end_data_region
; THUMB1-NOPIC: [[LABEL]]

; THUMB1-NOT: vld
; THUMB1: pop {{[^d]*$}}
