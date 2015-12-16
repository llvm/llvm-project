; RUN: llc -verify-machineinstrs -mtriple=i686-pc-windows-msvc < %s | FileCheck --check-prefix=X86 %s
; RUN: llc -verify-machineinstrs -mtriple=x86_64-pc-windows-msvc < %s | FileCheck --check-prefix=X64 %s

%struct.Dtor = type { i8 }

define void @simple_cleanup() #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %o = alloca %struct.Dtor, align 1
  invoke void @f(i32 1)
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  call x86_thiscallcc void @"\01??1Dtor@@QAE@XZ"(%struct.Dtor* %o) #2
  ret void

ehcleanup:                                        ; preds = %entry
  %0 = cleanuppad within none []
  call x86_thiscallcc void @"\01??1Dtor@@QAE@XZ"(%struct.Dtor* %o) #2 [ "funclet"(token %0) ]
  cleanupret from %0 unwind to caller
}

; CHECK: simple_cleanup:                         # @simple_cleanup
; CHECK:         pushq   %rbp
; CHECK:         subq    $48, %rsp
; CHECK:         leaq    48(%rsp), %rbp
; CHECK:         movq    $-2, -8(%rbp)
; CHECK:         movl    $1, %ecx
; CHECK:         callq   f
; CHECK:         callq   "??1Dtor@@QAE@XZ"
; CHECK:         nop
; CHECK:         addq    $48, %rsp
; CHECK:         popq    %rbp
; CHECK:         retq

; CHECK: "?dtor$2@?0?simple_cleanup@4HA":
; CHECK:         callq   "??1Dtor@@QAE@XZ"
; CHECK:         retq

; CHECK: $cppxdata$simple_cleanup:
; CHECK-NEXT:         .long   429065506
; CHECK-NEXT:         .long   1
; CHECK-NEXT:         .long   ($stateUnwindMap$simple_cleanup)@IMGREL
; CHECK-NEXT:         .long   0
; CHECK-NEXT:         .long   0
; CHECK-NEXT:         .long   3
; CHECK-NEXT:         .long   ($ip2state$simple_cleanup)@IMGREL
; UnwindHelp offset should match the -2 store above
; CHECK-NEXT:         .long   40
; CHECK-NEXT:         .long   0
; CHECK-NEXT:         .long   1

declare void @f(i32) #0

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind
declare x86_thiscallcc void @"\01??1Dtor@@QAE@XZ"(%struct.Dtor*) #1

define void @nested_cleanup() #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %o1 = alloca %struct.Dtor, align 1
  %o2 = alloca %struct.Dtor, align 1
  invoke void @f(i32 1)
          to label %invoke.cont unwind label %cleanup.outer

invoke.cont:                                      ; preds = %entry
  invoke void @f(i32 2)
          to label %invoke.cont.1 unwind label %cleanup.inner

invoke.cont.1:                                    ; preds = %invoke.cont
  call x86_thiscallcc void @"\01??1Dtor@@QAE@XZ"(%struct.Dtor* %o2) #2
  invoke void @f(i32 3)
          to label %invoke.cont.2 unwind label %cleanup.outer

invoke.cont.2:                                    ; preds = %invoke.cont.1
  call x86_thiscallcc void @"\01??1Dtor@@QAE@XZ"(%struct.Dtor* %o1) #2
  ret void

cleanup.inner:                                        ; preds = %invoke.cont
  %0 = cleanuppad within none []
  call x86_thiscallcc void @"\01??1Dtor@@QAE@XZ"(%struct.Dtor* %o2) #2 [ "funclet"(token %0) ]
  cleanupret from %0 unwind label %cleanup.outer

cleanup.outer:                                      ; preds = %invoke.cont.1, %cleanup.inner, %entry
  %1 = cleanuppad within none []
  call x86_thiscallcc void @"\01??1Dtor@@QAE@XZ"(%struct.Dtor* %o1) #2 [ "funclet"(token %1) ]
  cleanupret from %1 unwind to caller
}

; X86-LABEL: _nested_cleanup:
; X86: movl    $1, (%esp)
; X86: calll   _f
; X86: movl    $2, (%esp)
; X86: calll   _f
; X86: movl    $3, (%esp)
; X86: calll   _f

; X86: "?dtor$[[cleanup_inner:[0-9]+]]@?0?nested_cleanup@4HA":
; X86: LBB1_[[cleanup_inner]]: # %cleanup.inner{{$}}
; X86: pushl %ebp
; X86: leal    {{.*}}(%ebp), %ecx
; X86: calll   "??1Dtor@@QAE@XZ"
; X86: popl %ebp
; X86: retl

; X86: "?dtor$[[cleanup_outer:[0-9]+]]@?0?nested_cleanup@4HA":
; X86: LBB1_[[cleanup_outer]]: # %cleanup.outer{{$}}
; X86: pushl %ebp
; X86: leal    {{.*}}(%ebp), %ecx
; X86: calll   "??1Dtor@@QAE@XZ"
; X86: popl %ebp
; X86: retl

; X86: L__ehtable$nested_cleanup:
; X86:         .long   429065506
; X86:         .long   2
; X86:         .long   ($stateUnwindMap$nested_cleanup)
; X86:         .long   0
; X86:         .long   0
; X86:         .long   0
; X86:         .long   0
; X86:         .long   0
; X86:         .long   1
; X86: $stateUnwindMap$nested_cleanup:
; X86:         .long   -1
; X86:         .long   "?dtor$[[cleanup_outer]]@?0?nested_cleanup@4HA"
; X86:         .long   0
; X86:         .long   "?dtor$[[cleanup_inner]]@?0?nested_cleanup@4HA"

; X64-LABEL: nested_cleanup:
; X64: .Lfunc_begin1:
; X64: .Ltmp13:
; X64: movl    $1, %ecx
; X64: callq   f
; X64: .Ltmp15:
; X64: movl    $2, %ecx
; X64: callq   f
; X64: .Ltmp16:
; X64: callq   "??1Dtor@@QAE@XZ"
; X64: .Ltmp17:
; X64: movl    $3, %ecx
; X64: callq   f
; X64: .Ltmp18:

; X64: "?dtor$[[cleanup_inner:[0-9]+]]@?0?nested_cleanup@4HA":
; X64: LBB1_[[cleanup_inner]]: # %cleanup.inner{{$}}
; X64: pushq %rbp
; X64: leaq    {{.*}}(%rbp), %rcx
; X64: callq   "??1Dtor@@QAE@XZ"
; X64: popq %rbp
; X64: retq

; X64:        .seh_handlerdata
; X64:        .text
; X64:        .seh_endproc

; X64: "?dtor$[[cleanup_outer:[0-9]+]]@?0?nested_cleanup@4HA":
; X64: LBB1_[[cleanup_outer]]: # %cleanup.outer{{$}}
; X64: pushq %rbp
; X64: leaq    {{.*}}(%rbp), %rcx
; X64: callq   "??1Dtor@@QAE@XZ"
; X64: popq %rbp
; X64: retq

; X64:        .section .xdata,"dr"
; X64-NEXT: .align  4
; X64: $cppxdata$nested_cleanup:
; X64-NEXT: .long   429065506
; X64-NEXT: .long   2
; X64-NEXT: .long   ($stateUnwindMap$nested_cleanup)@IMGREL
; X64-NEXT: .long   0
; X64-NEXT: .long   0
; X64-NEXT: .long   5
; X64-NEXT: .long   ($ip2state$nested_cleanup)@IMGREL
; X64-NEXT: .long   56
; X64-NEXT: .long   0
; X64-NEXT: .long   1

; X64: $stateUnwindMap$nested_cleanup:
; X64-NEXT: .long   -1
; X64-NEXT: .long   "?dtor$[[cleanup_outer]]@?0?nested_cleanup@4HA"@IMGREL
; X64-NEXT: .long   0
; X64-NEXT: .long   "?dtor$[[cleanup_inner]]@?0?nested_cleanup@4HA"@IMGREL

; X64: $ip2state$nested_cleanup:
; X64-NEXT: .long   .Lfunc_begin1@IMGREL
; X64-NEXT: .long   -1
; X64-NEXT: .long   .Ltmp13@IMGREL
; X64-NEXT: .long   0
; X64-NEXT: .long   .Ltmp15@IMGREL
; X64-NEXT: .long   1
; X64-NEXT: .long   .Ltmp17@IMGREL
; X64-NEXT: .long   0
; X64-NEXT: .long   .Ltmp18@IMGREL+1
; X64-NEXT: .long   -1

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
