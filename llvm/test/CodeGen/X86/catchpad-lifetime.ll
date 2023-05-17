; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s --check-prefix=X64
; RUN: llc -mtriple=i686-windows-msvc < %s | FileCheck %s --check-prefix=X86

declare void @throw()

declare i32 @__CxxFrameHandler3(...)

declare void @llvm.trap()

define void @test1() personality ptr @__CxxFrameHandler3 {
entry:
  %alloca2 = alloca ptr, align 4
  %alloca1 = alloca ptr, align 4
  store volatile ptr null, ptr %alloca1
  invoke void @throw()
          to label %unreachable unwind label %catch.dispatch

; X64-LABEL: test1:
; X64: movq  $0, -8(%rbp)
; X64: callq throw

; X86-LABEL: _test1:
; X86: pushl   %ebp
; X86: movl    %esp, %ebp
; X86: pushl   %ebx
; X86: pushl   %edi
; X86: pushl   %esi
; X86: subl    $20, %esp

; X86: movl  $0, -32(%ebp)
; X86: calll _throw

catch.dispatch:                                   ; preds = %entry
  %cs = catchswitch within none [label %catch.pad] unwind to caller

catch.pad:                                        ; preds = %catch.dispatch
  %cp = catchpad within %cs [ptr null, i32 0, ptr %alloca1]
  %v = load volatile ptr, ptr %alloca1
  store volatile ptr null, ptr %alloca1
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %alloca1)
  call void @llvm.lifetime.start.p0(i64 4, ptr %alloca2)
  store volatile ptr null, ptr %alloca1
  call void @llvm.trap()
  unreachable

; X64-LABEL: "?catch$2@?0?test1@4HA"
; X64: movq  $0, -8(%rbp)
; X64: movq  $0, -8(%rbp)
; X64: ud2

; X86-LABEL: "?catch$2@?0?test1@4HA"
; X86: movl  $0, -32(%ebp)
; X86: movl  $0, -32(%ebp)
; X86: ud2

unreachable:                                      ; preds = %entry
  unreachable
}

; X64-LABEL: $cppxdata$test1:
; X64: .long   40                      # CatchObjOffset

; -20 is difference between the end of the EH reg node stack object and the
; catch object at EBP -32.
; X86-LABEL: L__ehtable$test1:
; X86: .long   -20                      # CatchObjOffset

define void @test2() personality ptr @__CxxFrameHandler3 {
entry:
  %alloca2 = alloca ptr, align 4
  %alloca1 = alloca ptr, align 4
  store volatile ptr null, ptr %alloca1
  invoke void @throw()
          to label %unreachable unwind label %catch.dispatch

; X64-LABEL: test2:
; X64: movq  $0, -16(%rbp)
; X64: callq throw

; X86-LABEL: _test2:
; X86: movl  $0, -32(%ebp)
; X86: calll _throw


catch.dispatch:                                   ; preds = %entry
  %cs = catchswitch within none [label %catch.pad] unwind to caller

catch.pad:                                        ; preds = %catch.dispatch
  %cp = catchpad within %cs [ptr null, i32 0, ptr null]
  store volatile ptr null, ptr %alloca1
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %alloca1)
  call void @llvm.lifetime.start.p0(i64 4, ptr %alloca2)
  store volatile ptr null, ptr %alloca1
  call void @llvm.trap()
  unreachable

; X64-LABEL: "?catch$2@?0?test2@4HA"
; X64: movq  $0, -16(%rbp)
; X64: movq  $0, -16(%rbp)
; X64: ud2

; X86-LABEL: "?catch$2@?0?test2@4HA"
; X86: movl  $0, -32(%ebp)
; X86: movl  $0, -32(%ebp)
; X86: ud2


unreachable:                                      ; preds = %entry
  unreachable
}

; X64-LABEL: $cppxdata$test2:
; X64: .long   0                       # CatchObjOffset


; X86-LABEL: L__ehtable$test2:
; X86: .long   0                       # CatchObjOffset


; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #0

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #0

attributes #0 = { argmemonly nounwind }
